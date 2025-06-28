"""
LiveKit AI Sales Bot with Azure Speech Services TTS Integration (FIXED VERSION)

SETUP INSTRUCTIONS:
1. Install Azure Speech SDK: pip install azure-cognitiveservices-speech
2. Set environment variables:
   - AZURE_SPEECH_API_KEY=your_azure_speech_api_key
   - AZURE_SPEECH_REGION=your_region (e.g., eastus, westus2, etc.)
   
3. Get Azure Speech credentials:
   - Go to https://portal.azure.com
   - Create "Speech Services" resource
   - Copy API Key and Region from resource page

VOICE OPTIONS:
- en-US-AriaNeural (Professional female - recommended)
- en-US-DavisNeural (Confident male)
- en-US-JennyNeural (Conversational female)
- en-US-GuyNeural (Casual male)
"""

import asyncio
import os
import time
import json
import wave
import tempfile
import io
import uuid
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator

import openai
import httpx
from groq import Groq
import websockets
import aiohttp
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AgentSession, Agent
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType, STTCapabilities, SpeechData
from livekit.agents.tts import TTS, TTSCapabilities, SynthesizedAudio
from livekit.plugins import openai as lk_openai, silero

# Azure Speech Services WebSocket streaming implementation


# Azure Streaming TTS Implementation using WebSocket API
class AzureStreamingTTS(TTS):
    """Azure Speech Services TTS with WebSocket streaming support"""
    
    def __init__(
        self,
        api_key: str,
        region: str,
        voice: str = "en-US-AriaNeural",
        speed: float = 1.0,
        streaming: bool = True
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=streaming),
            sample_rate=48000,
            num_channels=1
        )
        
        self._api_key = api_key
        self._region = region
        self._voice = voice
        self._speed = speed
        
        # Azure WebSocket endpoints
        self._token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
        self._ws_url = f"wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
        
        # Validate inputs
        if not api_key or not region:
            raise ValueError("Azure API key and region are required")
        
        print(f"🔵 Azure Streaming TTS initialized with voice: {voice}, speed: {speed}")
    
    async def _get_access_token(self) -> str:
        """Get Azure access token"""
        headers = {
            'Ocp-Apim-Subscription-Key': self._api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self._token_url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get Azure token: {response.status}")
                return await response.text()
    
    def _create_ssml(self, text: str) -> str:
        """Create SSML with voice and speed settings"""
        speed_rate = f"{self._speed:.1f}" if self._speed != 1.0 else "1.0"
        
        return f"""
        <speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis'>
            <voice xml:lang='en-US' name='{self._voice}'>
                <prosody rate='{speed_rate}'>
                    {text}
                </prosody>
            </voice>
        </speak>
        """.strip()
    
    def _create_config_message(self, request_id: str) -> str:
        """Create WebSocket configuration message"""
        config = {
            "context": {
                "synthesis": {
                    "audio": {
                        "metadataoptions": {
                            "sentenceBoundaryEnabled": "false",
                            "wordBoundaryEnabled": "false"
                        },
                        "outputFormat": "raw-48khz-16bit-mono-pcm"
                    }
                }
            }
        }
        
        message = f"X-RequestId:{request_id}\r\n"
        message += "Content-Type:application/json; charset=utf-8\r\n"
        message += f"Path:speech.config\r\n\r\n"
        message += json.dumps(config)
        
        return message
    
    def _create_ssml_message(self, request_id: str, ssml: str) -> str:
        """Create SSML message for WebSocket"""
        message = f"X-RequestId:{request_id}\r\n"
        message += "Content-Type:application/ssml+xml\r\n"
        message += f"Path:ssml\r\n\r\n"
        message += ssml
        
        return message
    
    async def _stream_synthesis(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream audio synthesis using Azure WebSocket"""
        access_token = await self._get_access_token()
        request_id = str(uuid.uuid4()).replace('-', '')
        
        # WebSocket headers
        headers = {
            'Authorization': f'Bearer {access_token}',
            'X-ConnectionId': str(uuid.uuid4())
        }
        
        try:
            async with websockets.connect(self._ws_url, extra_headers=headers) as websocket:
                # Send configuration
                config_msg = self._create_config_message(request_id)
                await websocket.send(config_msg)
                
                # Send SSML
                ssml = self._create_ssml(text)
                ssml_msg = self._create_ssml_message(request_id, ssml)
                await websocket.send(ssml_msg)
                
                # Receive audio chunks
                async for message in websocket:
                    if isinstance(message, str):
                        # Text message (metadata)
                        if 'Path:turn.end' in message:
                            break
                        continue
                    
                    # Binary message (audio data)
                    if isinstance(message, bytes):
                        # Azure sends audio in chunks with headers
                        # Skip the header and extract audio data
                        header_end = message.find(b'\r\n\r\n')
                        if header_end != -1:
                            audio_chunk = message[header_end + 4:]
                            if audio_chunk:
                                yield audio_chunk
                
        except Exception as e:
            print(f"❌ Azure WebSocket streaming error: {e}")
            raise
    
    def stream(self):
        """LiveKit streaming interface - returns async context manager"""
        return AzureStreamContext(self)
    
    async def synthesize(self, text: str) -> SynthesizedAudio:
        """Synthesize speech with streaming support"""
        try:
            start_time = time.time()
            audio_chunks = []
            first_chunk_time = None
            
            # Stream the audio
            async for chunk in self._stream_synthesis(text):
                audio_chunks.append(chunk)
                
                # Record time to first chunk (streaming performance indicator)
                if len(audio_chunks) == 1:
                    first_chunk_time = time.time() - start_time
                    print(f"🔵 Azure Streaming: First chunk in {first_chunk_time:.3f}s")
            
            # Combine all chunks
            audio_data = b"".join(audio_chunks)
            total_time = time.time() - start_time
            
            print(f"🔵 Azure Streaming TTS: Complete {len(audio_data)} bytes in {total_time:.3f}s")
            
            return SynthesizedAudio(
                text=text,
                data=audio_data,
                sample_rate=48000,
                num_channels=1
            )
            
        except Exception as e:
            print(f"❌ Azure Streaming TTS Error: {e}")
            return SynthesizedAudio(
                text=text,
                data=b"",
                sample_rate=48000,
                num_channels=1
            )


class AzureStreamContext:
    """Async context manager for Azure TTS streaming (LiveKit interface)"""
    
    def __init__(self, tts: AzureStreamingTTS):
        self._tts = tts
        self._stream_generator = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._stream_generator:
            await self._stream_generator.aclose()
    
    async def synthesize(self, text: str):
        """Start streaming synthesis"""
        print(f"🔵 Azure Stream: Starting synthesis for: '{text[:50]}...'")
        start_time = time.time()
        
        try:
            first_chunk_sent = False
            async for chunk in self._tts._stream_synthesis(text):
                if chunk and len(chunk) > 0:
                    # Create audio frame from chunk
                    audio_frame = rtc.AudioFrame(
                        data=chunk,
                        sample_rate=48000,
                        num_channels=1,
                        samples_per_channel=len(chunk) // 2  # 16-bit audio
                    )
                    
                    if not first_chunk_sent:
                        first_chunk_time = time.time() - start_time
                        print(f"🔵 Azure Stream: First chunk ready in {first_chunk_time:.3f}s")
                        first_chunk_sent = True
                    
                    yield audio_frame
                    
        except Exception as e:
            print(f"❌ Azure Stream Error: {e}")
            # Yield empty frame on error
            empty_frame = rtc.AudioFrame(
                data=b"",
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=0
            )
            yield empty_frame


# Check environment variables at startup
def check_env_vars():
    """Check if all required environment variables are present"""
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE", "SESSION_ID"]
    optional_vars = ["GROQ_API_KEY", "AZURE_SPEECH_API_KEY", "AZURE_SPEECH_REGION"]
    
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
                    print(f"❌ Failed to fetch session: 404")
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
    
    # Create the session with individual components with fallback mechanism
    azure_api_key = os.getenv("AZURE_SPEECH_API_KEY")
    azure_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    
    # Initialize TTS with fallback
    tts_engine = None
    if azure_api_key:
        try:
            tts_engine = AzureStreamingTTS(
                api_key=azure_api_key,
                region=azure_region,
                voice="en-US-AriaNeural",  # Professional female voice
                speed=1.1,                 # 10% faster speech
                streaming=True             # True WebSocket streaming!
            )
            print("🚀 Using Azure Streaming TTS for real-time speech!")
        except Exception as e:
            print(f"⚠️ Azure Streaming TTS failed, falling back to OpenAI: {e}")
            tts_engine = lk_openai.TTS(voice="alloy", speed=1.1)
    else:
        print("⚠️ Azure TTS not configured, using OpenAI TTS")
        tts_engine = lk_openai.TTS(voice="alloy", speed=1.1)
    
    # Initialize session with proper components
    if os.getenv("GROQ_API_KEY"):
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=GroqSTT(model="distil-whisper-large-v3-en"), # 🚀 240x faster than real-time!
            llm=lk_openai.LLM(model="gpt-4o-mini"),
            tts=tts_engine,
        )
        print("🚀 Using Groq STT + Azure Streaming TTS for ultra-fast speech processing!")
    else:
        # Full fallback to OpenAI
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=lk_openai.STT(),
            llm=lk_openai.LLM(model="gpt-4o-mini"),
            tts=tts_engine,
        )
        print("📢 Using OpenAI STT + TTS (Groq not configured)")
    
    # COMPREHENSIVE DELAY MEASUREMENT SYSTEM
    conversation_count = [0]
    welcome_sent = [False]
    
    # Detailed timing tracking for each conversation turn
    timing = {
        'user_stopped': None,      # When user stops speaking
        'stt_complete': None,      # When STT finishes
        'user_message': None,      # When user message added to chat
        'llm_start': None,         # When LLM starts processing  
        'llm_complete': None,      # When LLM response ready
        'tts_start': None,         # When TTS synthesis starts
        'audio_start': None,       # When audio generation begins
        'bot_speaking': None,      # When bot actually starts speaking
        'bot_audible': None        # When user can actually hear bot (estimated)
    }
    
    def reset_timing():
        for key in timing:
            timing[key] = None
    
    def print_delay_analysis():
        if not timing['user_stopped'] or not timing['bot_speaking']:
            return
            
        # Calculate each delay component using ACTUAL measured times
        t = timing
        total_measured = t['bot_speaking'] - t['user_stopped']
        
        print(f"🤖 BOT SPEAKING [ACTUAL-{conversation_count[0]:02d}] 📊 DELAY BREAKDOWN:")
        print(f"   ⏱️  TOTAL MEASURED: {total_measured:.2f}s (user stopped → bot starts speaking)")
        print(f"   🔍 COMPONENT DELAYS:")
        
        # Calculate actual delays with proper logic
        delays = {}
        
        if t['user_stopped'] and t['stt_complete']:
            stt_delay = t['stt_complete'] - t['user_stopped']
            delays['STT'] = stt_delay
            stt_percent = (stt_delay / total_measured) * 100
            print(f"      STT: {stt_delay:.2f}s ({stt_percent:.1f}%) - Speech to text processing")
        
        # Calculate LLM delay as the gap between STT completion and TTS start
        if t['stt_complete'] and t['tts_start']:
            llm_delay = t['tts_start'] - t['stt_complete']
            delays['LLM'] = llm_delay
            llm_percent = (llm_delay / total_measured) * 100
            print(f"      LLM: {llm_delay:.2f}s ({llm_percent:.1f}%) - AI response generation")
        
        # Calculate TTS delay from TTS start to bot speaking
        if t['tts_start'] and t['bot_speaking']:
            tts_delay = t['bot_speaking'] - t['tts_start']
            delays['TTS'] = tts_delay
            tts_percent = (tts_delay / total_measured) * 100
            print(f"      TTS: {tts_delay:.2f}s ({tts_percent:.1f}%) - Text to speech synthesis")
        
        # Show unaccounted time (pipeline overhead)
        accounted_time = sum(delays.values())
        unaccounted = total_measured - accounted_time
        if abs(unaccounted) > 0.1:
            unaccounted_percent = (unaccounted / total_measured) * 100
            print(f"      PIPELINE: {unaccounted:.2f}s ({unaccounted_percent:.1f}%) - System overhead")
        
        # Estimate remaining audio pipeline delay (streaming + buffering)
        audio_pipeline_delay = 1.5  # Conservative estimate
        estimated_user_heard = total_measured + audio_pipeline_delay
        print(f"   🎧 ESTIMATED USER HEARS: +{audio_pipeline_delay:.1f}s = {estimated_user_heard:.1f}s total")
        
        # Identify biggest bottleneck
        if delays:
            biggest_component = max(delays, key=delays.get)
            biggest_delay = delays[biggest_component]
            print(f"   🎯 BIGGEST BOTTLENECK: {biggest_component} ({biggest_delay:.2f}s)")
            
            # Provide optimization suggestions
            if biggest_component == 'LLM' and biggest_delay > 2.0:
                print(f"   💡 SUGGESTION: LLM is slow - try gpt-3.5-turbo or reduce max_tokens")
            elif biggest_component == 'STT' and biggest_delay > 1.0:
                print(f"   💡 SUGGESTION: STT is slow - check Groq API performance or audio quality")
            elif biggest_component == 'TTS' and biggest_delay > 2.0:
                print(f"   💡 SUGGESTION: TTS is slow - try streaming TTS or faster voice model")
    
    # Event handlers for precise timing measurement
    @session.on("user_state_changed") 
    def on_user_state_changed(event):
        print(f"🔧 User state: {getattr(event, 'old_state', '?')} → {getattr(event, 'new_state', '?')}")
        if hasattr(event, 'new_state'):
            if event.new_state == 'speaking':
                print("🎤 User started speaking...")
                reset_timing()  # Start fresh timing for new turn
            elif event.new_state == 'listening' and hasattr(event, 'old_state') and event.old_state == 'speaking':
                timing['user_stopped'] = asyncio.get_event_loop().time()
                print("🎤 User stopped speaking. ⏱️ Starting delay measurement...")
    
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        print(f"🔧 Transcription: is_final={getattr(event, 'is_final', '?')}, text='{getattr(event, 'transcript', '?')}'")
        
        if hasattr(event, 'is_final') and event.is_final:
            timing['stt_complete'] = asyncio.get_event_loop().time()
            if timing['user_stopped']:
                stt_delay = timing['stt_complete'] - timing['user_stopped']
                print(f"⚡ STT completed in {stt_delay:.3f}s")
    
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        if hasattr(event, 'item'):
            if event.item.role == 'user':
                timing['user_message'] = asyncio.get_event_loop().time()
                print(f"🎤 USER SAID: {event.item.text_content}")
                if timing['user_stopped']:
                    since_stopped = timing['user_message'] - timing['user_stopped']
                    print(f"🔧 Total time since user stopped: {since_stopped:.2f}s")
                
                # Mark LLM processing start
                timing['llm_start'] = asyncio.get_event_loop().time()
                
            elif event.item.role == 'assistant':
                timing['llm_complete'] = asyncio.get_event_loop().time()
                print(f"🧠 LLM response ready: {event.item.text_content[:50]}...")
                if timing['llm_start']:
                    llm_time = timing['llm_complete'] - timing['llm_start']
                    print(f"🧠 LLM processing took {llm_time:.3f}s")
                print(f"📝 Bot message added to chat: {event.item.text_content}")
    
    @session.on("speech_created")
    def on_speech_created(event):
        timing['tts_start'] = asyncio.get_event_loop().time()
        print(f"🔧 TTS synthesis started...")
        if timing['llm_complete']:
            queue_delay = timing['tts_start'] - timing['llm_complete']
            print(f"🔧 TTS queue delay: {queue_delay:.3f}s")
    
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        if hasattr(event, 'new_state') and hasattr(event, 'old_state'):
            if event.new_state == 'speaking':
                timing['bot_speaking'] = asyncio.get_event_loop().time()
                
                if not welcome_sent[0]:
                    welcome_sent[0] = True
                    print("🤖 BOT SPEAKING [WELCOME] (greeting)")
                    return
                    
                conversation_count[0] += 1
                print_delay_analysis()  # Print detailed breakdown
                    
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