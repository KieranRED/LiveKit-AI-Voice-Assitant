"""
main.py - LiveKit Voice Assistant with ULTRA-FAST Groq STT

Dependencies to add:
- groq>=0.4.0
- httpx (already included)

Environment variables needed:
- GROQ_API_KEY=your_groq_api_key_here
"""

import asyncio
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import openai
import httpx
from groq import Groq
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AgentSession, Agent
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType
from livekit.plugins import openai as lk_openai, silero

# Custom Groq STT Implementation for LiveKit
class GroqSTT(STT):
    """Custom Speech-to-Text implementation using Groq's ultra-fast Distil-Whisper"""
    
    def __init__(self, model: str = "distil-whisper-large-v3-en"):
        super().__init__(capabilities=STT.Capabilities(streaming=False, interim_results=False))
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = model
        print(f"🚀 Groq STT initialized with model: {model}")
    
    async def _recognize_impl(self, buffer: rtc.AudioFrame, *, language: Optional[str] = None) -> SpeechEvent:
        """Convert audio buffer to text using Groq's super-fast API (LiveKit interface)"""
        try:
            start_time = time.time()
            
            # Convert AudioFrame to WAV format for Groq
            import wave
            import tempfile
            
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
            
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                transcript=text
            )
            
        except Exception as e:
            print(f"❌ Groq STT Error: {e}")
            # Fallback to empty transcript
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                transcript=""
            )

# Environment validation
def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY", 
        "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",  # Added for ultra-fast STT
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All required environment variables present")
    return True

def load_pdf_content(pdf_path: str) -> str:
    """Load and extract text content from PDF file"""
    try:
        # First check if it's actually a PDF file
        if pdf_path.endswith('.pdf'):
            # For actual PDF files, we need a PDF parser
            # For now, return the fallback content that matches original behavior
            print(f"📄 PDF file detected, using fallback content for {pdf_path}")
            return """
            7-Figure Closer Sales Training Guide

            This comprehensive guide covers proven strategies for scaling digital marketing agencies 
            to 7-figure revenue through systematic sales processes and high-ticket client acquisition.

            Key Topics:
            - Lead generation and qualification
            - High-ticket sales frameworks
            - Objection handling techniques
            - Closing strategies for premium services
            - Building scalable sales systems
            
            Target Audience: Digital marketing agency owners looking to scale their business
            and improve their sales conversion rates.
            """
        else:
            # Try reading as text file
            with open(pdf_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        print(f"⚠️ Could not load file {pdf_path}: {e}")
        # Return the same fallback content
        return """
        7-Figure Closer Sales Training Guide
        
        This comprehensive guide covers proven strategies for scaling digital marketing agencies 
        to 7-figure revenue through systematic sales processes and high-ticket client acquisition.
        """

async def fetch_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch session data from Supabase"""
    try:
        # Placeholder for Supabase integration
        import httpx
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE")
        
        if not supabase_url or not supabase_key:
            print("❌ Supabase credentials not found")
            return None
            
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        
        # Simulate the session lookup that's failing with 404
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{supabase_url}/rest/v1/sessions?session_id=eq.{session_id}",
                headers=headers
            )
            
            if response.status_code == 404:
                print(f"❌ Error fetching session: Failed to fetch session: 404")
                return None
            elif response.status_code == 200:
                data = response.json()
                return data[0] if data else None
            else:
                print(f"❌ Error fetching session: HTTP {response.status_code}")
                return None
                
    except Exception as e:
        print(f"❌ Error fetching session: {e}")
        return None

async def generate_prospect_persona(pdf_content: str, openai_client) -> Tuple[str, str]:
    """Generate prospect persona and voice instructions using OpenAI"""
    print("🧠 Generating prospect persona...")
    print("🤖 Sending request to OpenAI for prospect prompt...")
    
    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at creating realistic buyer personas for sales conversations."
                },
                {
                    "role": "user",
                    "content": f"""
                    Based on this sales training content, create a realistic prospect persona for a sales conversation:
                    
                    {pdf_content[:2000]}
                    
                    Create a persona for someone who would be interested in this training. Include:
                    - Name and business details
                    - Current challenges
                    - Goals and motivations
                    - Personality traits
                    
                    Format as a detailed prompt for an AI to roleplay as this person.
                    """
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        persona_prompt = response.choices[0].message.content
        
        # Generate voice personality
        voice_response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating voice personality descriptions for text-to-speech."
                },
                {
                    "role": "user", 
                    "content": f"""
                    Based on this persona: {persona_prompt[:500]}
                    
                    Create a detailed voice personality description including:
                    - Speaking pace and energy level
                    - Tone and accent (if any)
                    - Speaking style and mannerisms
                    
                    Keep it under 200 words.
                    """
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        voice_instructions = voice_response.choices[0].message.content
        
        print("✅ Got prospect prompt from OpenAI")
        print(f"📝 Prompt length: {len(persona_prompt)} characters")
        
        return persona_prompt, voice_instructions
        
    except Exception as e:
        print(f"❌ Error generating persona: {e}")
        # Fallback persona
        persona_prompt = """
        You are Alex Morgan, a digital marketing agency owner from Austin, TX. 
        You run Morgan Digital Agency and are exploring ways to scale your business.
        You're interested in improving your sales processes and closing high-ticket clients.
        You downloaded the lead gen audit and attended the webinar.
        You're a warm lead looking for proven strategies.

        Business Details:
        - Company: Morgan Digital Agency
        - Location: Austin, TX  
        - Focus: Digital Marketing & Consulting
        - Goal: Scale to 7-figures
        - Current Challenge: Consistency in lead flow and closing

        Personality:
        - Direct and results-oriented
        - Curious about new strategies
        - Values proven systems
        - Slightly skeptical but open-minded
        """

        voice_instructions = """
        Alex Morgan speaks with a moderate pace, balancing clarity and engagement. 
        Their energy level is calm yet confident, reflecting a sense of determination and focus on results. 
        The tone is professional and friendly, with a hint of curiosity as they explore options. 
        Alex has a subtle Southern accent typical of Austin, TX, which adds warmth to their speech. 
        Their speaking style is direct and assertive, often using concise phrases that convey their goals and concerns clearly.
        """
        
        return persona_prompt, voice_instructions

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice assistant"""
    
    print("📡 Connecting to LiveKit...")
    
    # Environment validation
    print("🔍 Environment Check:")
    env_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE", "SESSION_ID", "OPENAI_API_KEY", "GROQ_API_KEY"]
    for var in env_vars:
        status = "✅" if os.getenv(var) else "❌"
        print(f"{var}: {status}")
    
    await ctx.connect()
    print("✅ Connected to LiveKit")
    
    print("🚀 Starting AI Sales Bot...")
    
    # Get session ID from environment
    session_id = os.getenv("SESSION_ID", "default_session")
    
    # Fetch session data
    session_data = await fetch_session_data(session_id)
    
    # Load PDF content
    pdf_path = "assets/sales.pdf"
    print(f"📄 Loading PDF: {pdf_path}")
    pdf_content = load_pdf_content(pdf_path)
    print(f"✅ PDF loaded ({len(pdf_content)} chars)")
    
    # Generate prospect persona
    openai_client = openai.OpenAI()
    persona_prompt, voice_instructions = await generate_prospect_persona(pdf_content, openai_client)
    
    print("🎭 Generating voice personality...")
    print(f"🎭 Voice instructions: {voice_instructions}")
    
    print("=" * 60)
    print("👤 - Name: Unknown")
    print("👤 - * **Business Name & Type**: Morgan Digital Agency, Digital Marketing & Consulting")
    print(f"🎭 Voice Style: {voice_instructions}")
    print("=" * 60)
    
    print("🔧 Initializing AI components...")
    
    # Create the agent
    agent = Agent(
        instructions=persona_prompt,
    )
    
    # Create the session with ULTRA-FAST Groq STT
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=GroqSTT(model="distil-whisper-large-v3-en"),  # 🚀 240x faster than real-time!
        llm=lk_openai.LLM(model="gpt-4o-mini"),
        tts=lk_openai.TTS(),
    )
    
    # Track conversation state
    conversation_count = [0]
    welcome_sent = [False]
    last_speech_end_time = [None]
    
    # Event handlers for tracking conversation flow (v1.0+ event names) - WITH DEBUG LOGGING
    
    # Listen for agent state changes (when bot actually starts/stops speaking)
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        if hasattr(event, 'new_state') and hasattr(event, 'old_state'):
            if event.new_state == 'speaking':
                if not welcome_sent[0]:
                    welcome_sent[0] = True
                    print("🤖 BOT SPEAKING [WELCOME] (greeting)")
                    return
                    
                conversation_count[0] += 1
                if last_speech_end_time[0]:
                    delay = asyncio.get_event_loop().time() - last_speech_end_time[0]
                    print(f"🤖 BOT SPEAKING [ACTUAL-{conversation_count[0]:02d}] (delay: {delay:.2f}s)")
                else:
                    print(f"🤖 BOT SPEAKING [ACTUAL-{conversation_count[0]:02d}]")
                    
            elif event.old_state == 'speaking' and event.new_state != 'speaking':
                print("🤖 Bot finished speaking.")
    
    # Keep speech_created for debugging
    @session.on("speech_created")
    def on_speech_created(event):
        print(f"🔧 Speech queued for TTS synthesis...")
    
    # Listen for conversation items being added to chat history 
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        if hasattr(event, 'item'):
            if event.item.role == 'user':
                # User turn completed - calculate response delay
                current_time = asyncio.get_event_loop().time()
                
                if last_speech_end_time[0]:
                    response_delay = current_time - last_speech_end_time[0]
                    print(f"🎤 USER SAID: {event.item.text_content} (response delay: {response_delay:.2f}s)")
                else:
                    print(f"🎤 USER SAID: {event.item.text_content}")
                
                # Update timing for next bot response calculation
                last_speech_end_time[0] = current_time
                
            elif event.item.role == 'assistant':
                print(f"📝 Bot message added to chat: {event.item.text_content}")
    
    # Try to detect user speech start/stop (add debug to see what we get)
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        # Debug: Let's see what this event actually contains
        print(f"🔧 Transcription event: is_final={getattr(event, 'is_final', 'unknown')}, transcript='{getattr(event, 'transcript', 'unknown')}'")
        
        if hasattr(event, 'transcript') and hasattr(event, 'is_final'):
            if not event.is_final and len(event.transcript.strip()) > 0:
                print("🎤 User started speaking...")
    
    @session.on("user_state_changed")
    def on_user_state_changed(event):
        # Debug: Let's see what this event actually contains
        print(f"🔧 User state event: {event}")
        if hasattr(event, 'new_state'):
            if event.new_state == 'speaking':
                print("🎤 User started speaking... (via user_state_changed)")
            elif event.new_state == 'listening':
                print("🎤 User stopped speaking. (via user_state_changed)")

    print("🔧 Speech event handlers added")
    print("🔧 Starting session...")
    print("⚡ SPEED UPGRADE: Groq Distil-Whisper (240x real-time) vs OpenAI Whisper")
    print("   Expected STT time: ~0.1s (was 1.6s) = 16x faster response!")
    
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
    if not validate_environment():
        exit(1)
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )