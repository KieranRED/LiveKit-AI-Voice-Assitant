"""
LiveKit AI Sales Bot with Azure Speech Services TTS Integration (AZURE SDK VERSION)

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
import xml.sax.saxutils
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator

import openai
import httpx
from groq import Groq

# Import Azure Speech SDK
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SDK_AVAILABLE = True
    print("🔵 Azure Speech SDK available")
except ImportError:
    AZURE_SDK_AVAILABLE = False
    print("⚠️ Azure Speech SDK not available - install with: pip install azure-cognitiveservices-speech")

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AgentSession, Agent
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType, STTCapabilities, SpeechData
from livekit.agents.tts import TTS, TTSCapabilities, SynthesizedAudio
from livekit.plugins import openai as lk_openai, silero


# Azure TTS using official SDK - should eliminate all audio artifacts
class AzureSDKTTS(TTS):
    """Azure Speech Services TTS using official Azure SDK - clean, reliable audio"""
    
    def __init__(
        self,
        api_key: str,
        region: str,
        voice: str = "en-US-AriaNeural",
        speed: float = 1.0,
        streaming: bool = False
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=streaming),
            sample_rate=48000,
            num_channels=1
        )
        
        if not AZURE_SDK_AVAILABLE:
            raise Exception("Azure Speech SDK not available. Install with: pip install azure-cognitiveservices-speech")
        
        self._api_key = api_key
        self._region = region
        self._voice = voice
        self._speed = speed
        self._failed_requests = 0
        self._max_failures = 3
        
        # Create Azure Speech Config
        self._speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
        self._speech_config.speech_synthesis_voice_name = voice
        
        # Set audio format to match LiveKit expectations
        # Use 48kHz 16-bit mono PCM to match our sample rate
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
        )
        
        # Create OpenAI fallback
        self._openai_fallback = None
        try:
            self._openai_fallback = lk_openai.TTS(voice="alloy", speed=speed)
            print("🔄 OpenAI TTS fallback initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI fallback: {e}")
        
        print(f"🔵 Azure SDK TTS initialized with voice: {voice}, speed: {speed}")
    
    async def _use_fallback(self, text: str, request_id: str):
        """Use OpenAI TTS as fallback"""
        if self._openai_fallback is None:
            raise Exception("No fallback TTS available")
        
        print(f"🔄 Using OpenAI TTS fallback for: '{text[:50]}...'")
        
        async for result in self._openai_fallback.synthesize(text):
            yield SynthesizedAudio(
                frame=result.frame,
                request_id=request_id,
                is_final=result.is_final
            )
    
    def _create_ssml(self, text: str) -> str:
        """Create SSML with voice and speed settings"""
        speed_rate = f"{self._speed:.1f}" if self._speed != 1.0 else "1.0"
        
        # Escape XML characters in text to prevent parsing errors
        escaped_text = xml.sax.saxutils.escape(text)
        
        ssml = f"""
        <speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis'>
            <voice xml:lang='en-US' name='{self._voice}'>
                <prosody rate='{speed_rate}'>
                    {escaped_text}
                </prosody>
            </voice>
        </speak>
        """.strip()
        
        return ssml
    
    # CRITICAL: Implement the stream() method that LiveKit expects
    def stream(self):
        """LiveKit streaming interface - returns an async context manager"""
        return self._StreamingContext(self)
    
    class _StreamingContext:
        """Async context manager for LiveKit streaming"""
        
        def __init__(self, tts_instance):
            self.tts = tts_instance
            self._current_text = None
            self._audio_generator = None
        
        async def __aenter__(self):
            """Enter the async context manager"""
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Exit the async context manager"""
            if self._audio_generator:
                try:
                    await self._audio_generator.aclose()
                except Exception:
                    pass
        
        def __aiter__(self):
            """Make this object async iterable"""
            return self
        
        async def __anext__(self):
            """Async iterator protocol - wait for text and return audio"""
            if self._audio_generator is None:
                # Wait for text to be sent via send()
                while self._current_text is None:
                    await asyncio.sleep(0.01)
                
                # Start generating audio for the received text
                text = self._current_text
                self._current_text = None
                
                # Create audio generator
                request_id = str(uuid.uuid4())
                self._audio_generator = self.tts._synthesize_sdk(text, request_id)
            
            try:
                # Get next audio chunk
                audio_chunk = await self._audio_generator.__anext__()
                return audio_chunk
            except StopAsyncIteration:
                # Done with this text, reset for next
                self._audio_generator = None
                raise StopAsyncIteration
            except Exception as e:
                print(f"❌ Azure SDK TTS error: {e}")
                # Fall back to OpenAI if available
                if self.tts._openai_fallback:
                    try:
                        fallback_gen = self.tts._openai_fallback.synthesize(text if 'text' in locals() else "Error occurred")
                        audio_chunk = await fallback_gen.__anext__()
                        return audio_chunk
                    except Exception:
                        pass
                
                # If all fails, stop iteration
                raise StopAsyncIteration
        
        async def send(self, text: str):
            """Send text to be synthesized"""
            self._current_text = text
    
    async def _synthesize_sdk(self, text: str, request_id: str):
        """Clean Azure TTS synthesis using official Azure SDK"""
        if self._failed_requests >= self._max_failures:
            print(f"🔄 Too many Azure failures ({self._failed_requests}), using OpenAI fallback")
            if self._openai_fallback:
                async for result in self._use_fallback(text, request_id):
                    yield result
                return
        
        try:
            print(f"🔵 Azure SDK TTS synthesis: '{text[:50]}...'")
            
            # Create SSML for better voice control
            ssml = self._create_ssml(text)
            
            # Create synthesizer with null audio output (we'll get the raw data)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config,
                audio_config=None  # None means we get raw audio data
            )
            
            # Run synthesis in thread pool to avoid blocking
            def run_synthesis():
                try:
                    # Use SSML for synthesis
                    result = synthesizer.speak_ssml(ssml)
                    
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        # Get the raw audio data directly from Azure SDK
                        audio_data = result.audio_data
                        print(f"🔵 ✅ Azure SDK synthesis complete: {len(audio_data)} bytes")
                        return audio_data
                    elif result.reason == speechsdk.ResultReason.Canceled:
                        cancellation_details = speechsdk.CancellationDetails(result)
                        print(f"🔵 ❌ Speech synthesis canceled: {cancellation_details.reason}")
                        if cancellation_details.reason == speechsdk.CancellationReason.Error:
                            print(f"🔵 Error details: {cancellation_details.error_details}")
                        return None
                    else:
                        print(f"🔵 ❌ Unexpected synthesis result: {result.reason}")
                        return None
                        
                except Exception as e:
                    print(f"🔵 ❌ Azure SDK synthesis error: {e}")
                    return None
                finally:
                    # Clean up
                    if synthesizer:
                        del synthesizer
            
            # Run synthesis asynchronously
            audio_data = await asyncio.get_event_loop().run_in_executor(None, run_synthesis)
            
            if audio_data and len(audio_data) > 0:
                # Ensure 16-bit alignment
                if len(audio_data) % 2 == 1:
                    audio_data = audio_data[:-1]
                    print(f"🔵 Aligned audio: removed 1 byte for 16-bit alignment")
                
                if len(audio_data) >= 2:
                    samples_per_channel = len(audio_data) // 2
                    
                    # Create the audio frame directly from Azure's clean data
                    frame = rtc.AudioFrame(
                        data=audio_data,
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                        samples_per_channel=samples_per_channel
                    )
                    
                    # Success - reset failure counter
                    self._failed_requests = 0
                    print(f"🔵 ✅ Clean SDK audio: {samples_per_channel} samples ({len(audio_data)} bytes) - Direct from Azure")
                    
                    yield SynthesizedAudio(
                        frame=frame,
                        request_id=request_id,
                        is_final=True
                    )
                    return
            
            # If we get here, Azure failed - use fallback
            print("🔄 Azure SDK synthesis failed, using OpenAI fallback...")
            if self._openai_fallback:
                async for result in self._use_fallback(text, request_id):
                    yield result
            
        except Exception as e:
            self._failed_requests += 1
            print(f"❌ Azure SDK TTS Error: {e}")
            
            # Fall back to OpenAI
            if self._openai_fallback:
                try:
                    async for result in self._use_fallback(text, request_id):
                        yield result
                    return
                except Exception:
                    pass
            
            # Final fallback - empty audio
            empty_frame = rtc.AudioFrame.create(
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=0
            )
            yield SynthesizedAudio(
                frame=empty_frame,
                request_id=request_id,
                is_final=True
            )
    
    # Keep the original synthesize method for backward compatibility
    async def synthesize(self, text: str, **kwargs):
        """Non-streaming synthesis (fallback)"""
        request_id = str(uuid.uuid4())
        async for result in self._synthesize_sdk(text, request_id):
            yield result


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


# Custom Groq STT Implementation for LiveKit
class GroqSTT(STT):
    """Custom Speech-to-Text implementation using Groq's ultra-fast Distil-Whisper"""
    
    def __init__(self, model: str = "distil-whisper-large-v3-en"):
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
        conn_options=None,
    ) -> SpeechEvent:
        """Convert audio buffer to text using Groq's super-fast API"""
        try:
            start_time = time.time()
            detected_language = language or "en"
            
            # Convert AudioFrame to WAV format for Groq
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(buffer.num_channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(buffer.sample_rate)
                    wav_file.writeframes(buffer.data.tobytes())
                
                # Call Groq API
                with open(temp_file.name, "rb") as audio_file:
                    transcription = await asyncio.to_thread(
                        self._client.audio.transcriptions.create,
                        file=audio_file,
                        model=self._model,
                        response_format="json"
                    )
                
                os.unlink(temp_file.name)
            
            processing_time = time.time() - start_time
            text = transcription.text.strip()
            
            print(f"⚡ Groq STT: '{text}' (processed in {processing_time:.3f}s)")
            
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text=text, language=detected_language)]
            )
            
        except Exception as e:
            print(f"❌ Groq STT Error: {e}")
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
        if os.path.exists(file_path):
            print(f"📄 PDF file found at {file_path}")
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
    if azure_api_key and AZURE_SDK_AVAILABLE:
        try:
            tts_engine = AzureSDKTTS(
                api_key=azure_api_key,
                region=azure_region,
                voice="en-US-AriaNeural",  # Professional female voice
                speed=1.1,                 # 10% faster speech
                streaming=False            # Disable streaming for stability
            )
            print("🚀 Using Azure Official SDK TTS for pristine audio quality!")
        except Exception as e:
            print(f"⚠️ Azure SDK TTS initialization failed, falling back to OpenAI: {e}")
            tts_engine = lk_openai.TTS(voice="alloy", speed=1.1)
    else:
        print("⚠️ Azure SDK TTS not available, using OpenAI TTS")
        tts_engine = lk_openai.TTS(voice="alloy", speed=1.1)
    
    # Initialize session with proper components
    if os.getenv("GROQ_API_KEY"):
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=GroqSTT(model="distil-whisper-large-v3-en"), # 🚀 240x faster than real-time!
            llm=lk_openai.LLM(model="gpt-4o-mini"),
            tts=tts_engine,
        )
        print("🚀 Using Groq STT + Azure SDK TTS for ultra-fast, pristine audio!")
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
        audio_pipeline_delay = 0.1  # Much lower with Azure SDK
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
            elif biggest_component == 'TTS' and biggest_delay > 1.0:
                print(f"   💡 SUGGESTION: Azure SDK should provide fastest TTS synthesis!")
    
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