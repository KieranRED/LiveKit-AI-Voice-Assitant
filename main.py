import asyncio
import os
import time
import json
import wave
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import openai
import httpx
from groq import Groq
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AgentSession, Agent
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType, STTCapabilities, SpeechData
from livekit.plugins import openai as lk_openai, silero


# Check environment variables at startup
def check_env_vars():
    """Check if all required environment variables are present"""
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE", "SESSION_ID"]
    optional_vars = ["GROQ_API_KEY"]
    
    print("🔍 Environment Check:")
    all_good = True
    
    for var in required_vars:
        if os.getenv(var):
            print(f"{var}: ✅")
        else:
            print(f"{var}: ❌ Missing")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"{var}: ✅")
        else:
            print(f"{var}: ⚠️ Optional (will use fallback)")
    
    if not all_good:
        print("❌ Missing environment variables:", ", ".join([var for var in required_vars if not os.getenv(var)]))
        return False
    else:
        print("✅ All required environment variables present")
        return True


# Custom Groq STT Implementation for LiveKit (FIXED VERSION)
class GroqSTT(STT):
    """Custom Speech-to-Text implementation using Groq's ultra-fast Distil-Whisper"""
    
    def __init__(self, model: str = "distil-whisper-large-v3-en"):
        # Proper initialization with capabilities
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False)
        )
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = model
        print(f"🚀 Groq STT initialized with model: {model}")
    
    async def _recognize_impl(
        self, 
        buffer: rtc.AudioFrame, 
        *, 
        language: Optional[str] = None,
        conn_options=None,  # Add this parameter that LiveKit passes
    ) -> SpeechEvent:
        """Convert audio buffer to text using Groq's super-fast API (LiveKit interface)"""
        try:
            start_time = time.time()
            
            # Default language if not provided
            detected_language = language or "en"
            
            # Convert AudioFrame to WAV format for Groq
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Create WAV file with proper format
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(buffer.num_channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(buffer.sample_rate)
                    wav_file.writeframes(buffer.data.tobytes())
                
                # Call Groq API - this is where the magic happens!
                with open(temp_file.name, "rb") as audio_file:
                    transcription = await asyncio.to_thread(
                        self._client.audio.transcriptions.create,
                        file=audio_file,
                        model=self._model,
                        response_format="json"
                    )
                
                # Clean up temp file
                os.unlink(temp_file.name)
            
            processing_time = time.time() - start_time
            text = transcription.text.strip()
            
            print(f"⚡ Groq STT: '{text}' (processed in {processing_time:.3f}s)")
            
            # Return successful transcription
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text=text, language=detected_language)]
            )
            
        except Exception as e:
            print(f"❌ Groq STT Error: {e}")
            # Return empty result on error
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="", language=language or "en")]
            )


# Database operations for session management
async def get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch session data from Supabase"""
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE")
    
    if not supabase_url or not service_role_key:
        print("❌ Missing Supabase credentials")
        return None
    
    url = f"{supabase_url}/rest/v1/sessions?id=eq.{session_id}"
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    print(f"✅ Session data loaded for: {session_id}")
                    return data[0]
                else:
                    print(f"❌ No session found for: {session_id}")
                    return None
            else:
                print(f"❌ Failed to fetch session: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"❌ Error fetching session: {e}")
        return None


async def load_pdf_content(file_path: str = "assets/sales.pdf") -> str:
    """Load and extract text from PDF file"""
    print(f"📄 Loading PDF: {file_path}")
    
    try:
        # Try to read the PDF file
        if os.path.exists(file_path):
            print(f"📄 PDF file found at {file_path}")
            # For now, return fallback content since we don't have PyPDF2
            # In production, you'd use: pip install PyPDF2
            print(f"📄 PDF file detected, using fallback content for {file_path}")
            fallback_content = """
            Morgan Digital Agency - Sales Presentation
            
            We help small to medium-sized businesses grow through digital marketing strategies:
            - Search Engine Optimization (SEO)
            - Pay-Per-Click Advertising (PPC) 
            - Social Media Marketing
            - Email Marketing Campaigns
            - Website Design & Development
            - Content Marketing
            - Analytics & Reporting
            
            Our proven track record includes helping over 200+ businesses increase their online presence and revenue.
            
            Key Benefits:
            - Increase website traffic by 150%+ 
            - Generate more qualified leads
            - Improve conversion rates
            - Build brand awareness
            - Track ROI with detailed analytics
            
            We offer customized solutions based on your business goals and budget.
            """
            return fallback_content.strip()
        else:
            print(f"❌ PDF file not found: {file_path}")
            return "Sales presentation content not available."
            
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return "Sales presentation content not available."


async def generate_prospect_prompt(session_data: Optional[Dict[str, Any]], pdf_content: str) -> str:
    """Generate a dynamic prompt based on session data and PDF content"""
    print("🧠 Generating prospect persona...")
    
    # Default prospect if no session data
    default_prospect = {
        "name": "Unknown",
        "business_info": "* **Business Name & Type**: Morgan Digital Agency, Digital Marketing & Consulting",
        "business_size": "Small to medium-sized business (10-50 employees)",
        "industry": "Digital Marketing & Consulting",
        "location": "Austin, Texas",
        "pain_points": [
            "Need to increase online presence",
            "Looking for better lead generation",
            "Want to improve conversion rates",
            "Seeking ROI tracking and analytics"
        ],
        "goals": [
            "Grow website traffic", 
            "Generate more qualified leads",
            "Increase brand awareness",
            "Improve customer engagement"
        ]
    }
    
    prospect = default_prospect
    if session_data:
        print(f"📊 Using session data for prospect: {session_data.get('name', 'Unknown')}")
        prospect = {
            "name": session_data.get("name", "Unknown"),
            "business_info": session_data.get("business_info", default_prospect["business_info"]),
            "business_size": session_data.get("business_size", default_prospect["business_size"]),
            "industry": session_data.get("industry", default_prospect["industry"]),
            "location": session_data.get("location", default_prospect["location"]),
            "pain_points": session_data.get("pain_points", default_prospect["pain_points"]),
            "goals": session_data.get("goals", default_prospect["goals"])
        }
    
    # Generate voice personality using OpenAI
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    personality_prompt = f"""Create a detailed voice personality description for a sales representative from Morgan Digital Agency who will be speaking with this prospect:

Prospect Profile:
- Name: {prospect['name']}
- Business: {prospect['business_info']}
- Size: {prospect['business_size']}
- Industry: {prospect['industry']}
- Location: {prospect['location']}
- Pain Points: {', '.join(prospect['pain_points'])}
- Goals: {', '.join(prospect['goals'])}

Based on this prospect, describe the ideal voice personality for the sales rep including:
- Speaking pace and energy level
- Tone and accent (considering location/industry)
- Speaking style and mannerisms

Keep it concise but detailed. This will guide how the AI speaks to this specific prospect."""

    print("🤖 Sending request to OpenAI for prospect prompt...")
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert sales trainer who creates voice personality profiles for sales representatives."},
                {"role": "user", "content": personality_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        voice_personality = response.choices[0].message.content
        print("✅ Got prospect prompt from OpenAI")
        
    except Exception as e:
        print(f"❌ Error generating voice personality: {e}")
        voice_personality = "Speak with a professional, friendly tone. Be enthusiastic about digital marketing solutions while remaining conversational and approachable."
    
    # Build the complete prompt
    prompt = f"""You are a skilled sales representative from Morgan Digital Agency speaking with a prospect. Here's what you know:

PROSPECT INFORMATION:
- Name: {prospect['name']}
- {prospect['business_info']}
- Business Size: {prospect['business_size']}
- Industry: {prospect['industry']}
- Location: {prospect['location']}
- Pain Points: {', '.join(prospect['pain_points'])}
- Goals: {', '.join(prospect['goals'])}

COMPANY INFORMATION:
{pdf_content}

VOICE PERSONALITY:
{voice_personality}

YOUR ROLE:
- Build rapport and understand their specific needs
- Present relevant solutions from your service offerings
- Ask qualifying questions to uncover opportunities
- Share success stories and case studies when relevant
- Guide toward next steps (consultation, proposal, etc.)
- Be consultative, not pushy

CONVERSATION GUIDELINES:
- Start with a warm greeting and introduction
- Ask open-ended questions about their business
- Listen actively and respond to their specific needs
- Present solutions that directly address their pain points
- Use their name naturally in conversation
- Be authentic and build trust
- Focus on value, not just features

Keep responses conversational (2-3 sentences) unless they ask for detailed information."""

    print(f"📝 Prompt length: {len(prompt)} characters")
    return prompt, voice_personality


# Main entrypoint function
async def entrypoint(ctx: JobContext):
    """Main entry point for the voice agent"""
    print("📡 Connecting to LiveKit...")
    
    # Environment check
    if not check_env_vars():
        return
    
    # Connect to the LiveKit room
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    print("✅ Connected to LiveKit")
    print("🚀 Starting AI Sales Bot...")
    
    # Get session data
    session_id = os.getenv("SESSION_ID", "default")
    session_data = await get_session_data(session_id)
    
    # Load sales content
    pdf_content = await load_pdf_content()
    print(f"✅ PDF loaded ({len(pdf_content)} chars)")
    
    # Generate dynamic prompt and voice personality
    prompt, voice_instructions = await generate_prospect_prompt(session_data, pdf_content)
    
    print("🎭 Generating voice personality...")
    print(f"🎭 Voice instructions: {voice_instructions}")
    print("=" * 60)
    if session_data:
        print(f"👤 - Name: {session_data.get('name', 'Unknown')}")
        print(f"👤 - {session_data.get('business_info', 'Business info not available')}")
    else:
        print("👤 - Name: Unknown")
        print("👤 - * **Business Name & Type**: Morgan Digital Agency, Digital Marketing & Consulting")
    print(f"🎭 Voice Style: {voice_instructions}")
    print("=" * 60)
    
    print("🔧 Initializing AI components...")
    
    # Create the agent with instructions
    agent = Agent(
        instructions=prompt,
    )
    
    # Create the session with individual components
    if os.getenv("GROQ_API_KEY"):
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=GroqSTT(model="distil-whisper-large-v3-en"), # 🚀 240x faster than real-time!
            llm=lk_openai.LLM(model="gpt-4o-mini"),
            tts=lk_openai.TTS(),
        )
        print("🚀 Using Groq STT for ultra-fast speech recognition!")
    else:
        # Fallback to OpenAI STT
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=lk_openai.STT(),
            llm=lk_openai.LLM(model="gpt-4o-mini"),
            tts=lk_openai.TTS(),
        )
        print("📢 Using OpenAI STT (slower fallback)")
    
    # Track conversation state for ACCURATE delay measurement
    conversation_count = [0]
    welcome_sent = [False]
    user_stopped_speaking_time = [None]  # When user actually stopped talking
    user_said_time = [None]             # When transcription completed
    llm_start_time = [None]             # When LLM starts processing
    tts_start_time = [None]             # When TTS starts
    
    # Event handlers for tracking REAL conversation delays
    
    # Track when user stops speaking (start of delay measurement)
    @session.on("user_state_changed") 
    def on_user_state_changed(event):
        print(f"🔧 User state: {getattr(event, 'old_state', '?')} → {getattr(event, 'new_state', '?')}")
        if hasattr(event, 'new_state'):
            if event.new_state == 'speaking':
                print("🎤 User started speaking...")
            elif event.new_state == 'listening' and hasattr(event, 'old_state') and event.old_state == 'speaking':
                user_stopped_speaking_time[0] = asyncio.get_event_loop().time()
                print("🎤 User stopped speaking. ⏱️ Starting delay timer...")
    
    # Track transcription completion
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        print(f"🔧 Transcription: is_final={getattr(event, 'is_final', '?')}, text='{getattr(event, 'transcript', '?')}'")
        
        if hasattr(event, 'is_final') and event.is_final:
            user_said_time[0] = asyncio.get_event_loop().time()
            if user_stopped_speaking_time[0]:
                stt_delay = user_said_time[0] - user_stopped_speaking_time[0]
                print(f"⚡ STT completed in {stt_delay:.3f}s")
    
    # Track conversation items (when LLM processes)
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        if hasattr(event, 'item'):
            if event.item.role == 'user':
                print(f"🎤 USER SAID: {event.item.text_content}")
                if user_said_time[0] and user_stopped_speaking_time[0]:
                    since_stopped = asyncio.get_event_loop().time() - user_stopped_speaking_time[0]
                    print(f"🔧 Total time since user stopped: {since_stopped:.2f}s")
                
            elif event.item.role == 'assistant':
                llm_start_time[0] = asyncio.get_event_loop().time()
                print(f"🧠 LLM response ready: {event.item.text_content[:50]}...")
                if user_said_time[0]:
                    llm_delay = llm_start_time[0] - user_said_time[0]
                    print(f"🧠 LLM processing took {llm_delay:.3f}s")
    
    # Track TTS start
    @session.on("speech_created")
    def on_speech_created(event):
        tts_start_time[0] = asyncio.get_event_loop().time()
        print(f"🔧 TTS synthesis started...")
        if llm_start_time[0]:
            tts_queue_delay = tts_start_time[0] - llm_start_time[0]
            print(f"🔧 TTS queue delay: {tts_queue_delay:.3f}s")
    
    # Track when bot actually starts speaking (most important metric)
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        if hasattr(event, 'new_state') and hasattr(event, 'old_state'):
            if event.new_state == 'speaking':
                current_time = asyncio.get_event_loop().time()
                
                if not welcome_sent[0]:
                    welcome_sent[0] = True
                    print("🤖 BOT SPEAKING [WELCOME] (greeting)")
                    return
                    
                conversation_count[0] += 1
                
                # Calculate delay from user stopping to bot speaking (TTS start)
                if user_stopped_speaking_time[0]:
                    tts_start_delay = current_time - user_stopped_speaking_time[0]
                    
                    # Add estimated audio pipeline delay (streaming + buffering + playback)
                    estimated_audio_delay = 2.0  # Conservative estimate for real user experience
                    total_user_delay = tts_start_delay + estimated_audio_delay
                    
                    print(f"🤖 BOT SPEAKING [ACTUAL-{conversation_count[0]:02d}]")
                    print(f"   🎯 TTS START DELAY: {tts_start_delay:.2f}s")
                    print(f"   🎧 ESTIMATED USER-HEARD DELAY: {total_user_delay:.2f}s")
                    
                    # Breakdown of delays
                    if user_said_time[0] and tts_start_time[0]:
                        stt_time = user_said_time[0] - user_stopped_speaking_time[0] 
                        # LLM time is from STT completion to TTS start
                        llm_time = tts_start_time[0] - user_said_time[0] if tts_start_time[0] > user_said_time[0] else 0
                        # TTS generation time (from TTS start to agent speaking)
                        tts_gen_time = current_time - tts_start_time[0] if current_time > tts_start_time[0] else 0
                        
                        print(f"   📊 Breakdown: STT={stt_time:.2f}s + LLM={llm_time:.2f}s + TTS={tts_gen_time:.2f}s + Audio≈{estimated_audio_delay:.1f}s")
                else:
                    print(f"🤖 BOT SPEAKING [ACTUAL-{conversation_count[0]:02d}]")
                    
            elif event.old_state == 'speaking' and event.new_state != 'speaking':
                print("🤖 Bot finished speaking. Ready for next input...")

    print("🔧 Speech event handlers added")
    print("✅ AI session initialized successfully")
    print("🎤 Listening for user input...")
    
    # Start the session
    await session.start(agent=agent, room=ctx.room)
    
    # Send welcome message
    print("🗣️ Sending welcome message...")
    await session.say("Hey! Can you hear me clearly?", allow_interruptions=True)
    
    print("🎉 Sales bot ready! Conversation active...")
    
    # Keep the session alive
    while True:
        await asyncio.sleep(30)
        print("💓 Bot running...")


if __name__ == "__main__":
    # Configure worker options
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        worker_type=agents.WorkerType.ROOM,
    )
    
    print("🤖 Starting LiveKit AI Sales Bot Worker...")
    cli.run_app(worker_options)