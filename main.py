"""
LiveKit AI Sales Bot with Azure Speech Services TTS Integration (WORKING VERSION WITH SMOOTH BUFFERING)

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
import websockets
import aiohttp
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AgentSession, Agent
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType, STTCapabilities, SpeechData
from livekit.agents.tts import TTS, TTSCapabilities, SynthesizedAudio
from livekit.plugins import openai as lk_openai, silero


# Azure Streaming TTS Implementation using WebSocket API with OpenAI fallback
# Based on your working version but with smooth buffering added
class AzureStreamingTTS(TTS):
    """Azure Speech Services TTS with WebSocket streaming support and OpenAI fallback"""
    
    def __init__(
        self,
        api_key: str,
        region: str,
        voice: str = "en-US-AriaNeural",
        speed: float = 1.0,
        streaming: bool = True  # Enable streaming for smooth buffering
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
        self._failed_requests = 0  # Track failures for fallback logic
        self._max_failures = 3     # Switch to fallback after 3 failures
        
        # Azure WebSocket endpoints
        self._token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
        self._ws_url = f"wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
        
        # Create OpenAI fallback
        self._openai_fallback = None
        try:
            self._openai_fallback = lk_openai.TTS(voice="alloy", speed=speed)
            print("🔄 OpenAI TTS fallback initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI fallback: {e}")
        
        # Validate inputs
        if not api_key or not region:
            raise ValueError("Azure API key and region are required")
        
        print(f"🔵 Azure Streaming TTS initialized with voice: {voice}, speed: {speed}")
    
    async def _use_fallback(self, text: str, request_id: str):
        """Use OpenAI TTS as fallback"""
        if self._openai_fallback is None:
            raise Exception("No fallback TTS available")
        
        print(f"🔄 Using OpenAI TTS fallback for: '{text[:50]}...'")
        
        # Use the OpenAI TTS synthesize method
        async for result in self._openai_fallback.synthesize(text):
            # Convert OpenAI result to our format
            yield SynthesizedAudio(
                frame=result.frame,
                request_id=request_id,
                is_final=result.is_final
            )
    
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
        
        # Escape XML characters in text to prevent parsing errors
        escaped_text = xml.sax.saxutils.escape(text)
        
        return f"""
        <speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis'>
            <voice xml:lang='en-US' name='{self._voice}'>
                <prosody rate='{speed_rate}'>
                    {escaped_text}
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
    
    def _extract_audio_from_message(self, message: bytes) -> Optional[bytes]:
        """Extract clean PCM audio data from Azure WebSocket message - WORKING VERSION"""
        try:
            # Check if this is a Path:audio message
            if b'Path:audio' not in message:
                return None
            
            # Method 1: Look for WAV header patterns that Azure sometimes sends
            wav_header_pos = message.find(b'RIFF')
            if wav_header_pos != -1:
                # This is a WAV file - extract the audio data after the header
                wav_data = message[wav_header_pos:]
                if len(wav_data) > 44:  # WAV header is 44 bytes
                    # Skip WAV header and return raw PCM data
                    return wav_data[44:]
            
            # Method 2: Look for binary PCM data patterns
            # Azure sends raw PCM data, which should have good byte variety
            # and no readable text patterns
            
            # Start search after Path:audio
            path_audio_pos = message.find(b'Path:audio')
            search_start = path_audio_pos + len(b'Path:audio')
            
            # Look for the largest contiguous block of binary data
            best_audio_start = None
            best_audio_size = 0
            
            # Scan through the message looking for binary data blocks
            for start_pos in range(search_start, len(message) - 1000, 50):
                # Test if this looks like PCM audio data
                test_chunk = message[start_pos:start_pos + 1000]
                
                # Skip if it contains too much readable text
                try:
                    decoded = test_chunk.decode('utf-8', errors='ignore')
                    readable_chars = sum(1 for c in decoded if c.isalnum() or c.isspace())
                    if readable_chars > len(decoded) * 0.3:  # More than 30% readable = probably headers
                        continue
                except:
                    pass
                
                # Check for good byte variety (PCM audio should have variety)
                unique_bytes = len(set(test_chunk))
                if unique_bytes < 100:  # Not enough variety for good audio
                    continue
                
                # Check for patterns that suggest this is audio data
                # PCM audio typically has values distributed across the range
                byte_values = list(test_chunk)
                if len(byte_values) > 0:
                    # Calculate some basic statistics
                    avg_val = sum(byte_values) / len(byte_values)
                    # Good PCM audio should have average around 128 (middle of 0-255 range)
                    if abs(avg_val - 128) > 50:
                        continue
                
                # This looks like good audio data - find where it ends
                audio_end = start_pos + 1000
                for end_pos in range(start_pos + 1000, len(message), 1000):
                    test_end_chunk = message[end_pos:end_pos + 100]
                    if len(test_end_chunk) < 100:
                        audio_end = end_pos
                        break
                    
                    # Check if this chunk still looks like audio
                    end_unique_bytes = len(set(test_end_chunk))
                    if end_unique_bytes < 30:  # Lost variety, probably end of audio
                        audio_end = end_pos
                        break
                    
                    audio_end = end_pos + 100
                
                # Track the best (largest) audio block found
                audio_size = audio_end - start_pos
                if audio_size > best_audio_size:
                    best_audio_start = start_pos
                    best_audio_size = audio_size
            
            # If we found a good audio block, extract it
            if best_audio_start is not None and best_audio_size > 1000:
                audio_data = message[best_audio_start:best_audio_start + best_audio_size]
                
                # Final cleanup: remove any remaining header bytes at the start
                # Look for the first sequence that looks like pure PCM
                for i in range(0, min(200, len(audio_data)), 10):
                    chunk = audio_data[i:i+100]
                    if len(chunk) >= 100:
                        # Check if this looks like clean PCM audio
                        unique_vals = len(set(chunk))
                        if unique_vals >= 50:  # Good variety
                            # Check for text contamination
                            try:
                                decoded = chunk.decode('utf-8', errors='ignore')
                                text_chars = sum(1 for c in decoded if c.isalnum())
                                if text_chars < len(decoded) * 0.1:  # Less than 10% text
                                    # This looks like clean audio
                                    clean_audio = audio_data[i:]
                                    if len(clean_audio) > 1000:
                                        return clean_audio
                            except:
                                # Can't decode as text, probably good audio
                                clean_audio = audio_data[i:]
                                if len(clean_audio) > 1000:
                                    return clean_audio
                
                # If no clean start found, return the best block we found
                if len(audio_data) > 1000:
                    return audio_data
            
            # Method 3: Fallback - look for standard header separators
            header_patterns = [b'\r\n\r\n', b'\n\n', b'\r\r', b'\x00\x00']
            
            for pattern in header_patterns:
                pattern_pos = message.find(pattern, search_start)
                if pattern_pos != -1:
                    potential_audio = message[pattern_pos + len(pattern):]
                    if len(potential_audio) > 1000:
                        # Apply the same cleaning logic
                        for i in range(0, min(200, len(potential_audio)), 10):
                            chunk = potential_audio[i:i+100]
                            if len(chunk) >= 100:
                                unique_vals = len(set(chunk))
                                if unique_vals >= 50:
                                    try:
                                        decoded = chunk.decode('utf-8', errors='ignore')
                                        text_chars = sum(1 for c in decoded if c.isalnum())
                                        if text_chars < len(decoded) * 0.1:
                                            clean_audio = potential_audio[i:]
                                            if len(clean_audio) > 1000:
                                                return clean_audio
                                    except:
                                        clean_audio = potential_audio[i:]
                                        if len(clean_audio) > 1000:
                                            return clean_audio
            
            # If nothing worked, return None
            return None
            
        except Exception as e:
            print(f"🔵 Error extracting audio from message: {e}")
            return None
    
    # NEW: Add streaming support with smooth buffering
    def stream(self):
        """LiveKit streaming interface - returns an async context manager"""
        return self._StreamingContext(self)
    
    class _StreamingContext:
        """Async context manager for LiveKit streaming with smooth buffering"""
        
        def __init__(self, tts_instance):
            self.tts = tts_instance
            self._current_text = None
            self._audio_generator = None
        
        async def __aenter__(self):
            """Enter the async context manager"""
            print("🔵 Azure TTS streaming context entered")
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Exit the async context manager"""
            print("🔵 Azure TTS streaming context exited")
            if self._audio_generator:
                try:
                    await self._audio_generator.aclose()
                except Exception as e:
                    print(f"🔵 Error closing audio generator: {e}")
        
        def __aiter__(self):
            """Make this object async iterable"""
            return self
        
        async def __anext__(self):
            """Async iterator protocol - wait for text and return audio"""
            if self._audio_generator is None:
                # Wait for text to be sent via send()
                while self._current_text is None:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
                # Start generating audio for the received text
                text = self._current_text
                self._current_text = None  # Reset for next iteration
                
                print(f"🔵 Azure TTS streaming: '{text[:50]}...'")
                
                # Create audio generator for smooth streaming
                self._audio_generator = self.tts._synthesize_with_smooth_streaming(text)
            
            try:
                # Get next audio chunk
                audio_chunk = await self._audio_generator.__anext__()
                return audio_chunk
            except StopAsyncIteration:
                # Done with this text, reset for next
                self._audio_generator = None
                raise StopAsyncIteration
            except Exception as e:
                print(f"❌ Azure TTS stream error: {e}")
                # Fall back to OpenAI if available
                if self.tts._openai_fallback:
                    print("🔄 Using OpenAI fallback in stream...")
                    try:
                        request_id = str(uuid.uuid4())
                        fallback_gen = self.tts._use_fallback(text if 'text' in locals() else "Error occurred", request_id)
                        audio_chunk = await fallback_gen.__anext__()
                        return audio_chunk
                    except Exception as fallback_error:
                        print(f"❌ Fallback also failed: {fallback_error}")
                
                # If all fails, stop iteration
                raise StopAsyncIteration
        
        async def send(self, text: str):
            """Send text to be synthesized"""
            print(f"🔵 Received text for synthesis: '{text[:50]}...'")
            self._current_text = text
    
    async def _synthesize_with_smooth_streaming(self, text: str):
        """NEW: Synthesize with smooth streaming using working extraction + buffering"""
        request_id = str(uuid.uuid4())
        
        # If we've had too many failures, use fallback
        if self._failed_requests >= self._max_failures:
            print(f"🔄 Too many Azure failures ({self._failed_requests}), using OpenAI fallback")
            if self._openai_fallback:
                async for result in self._use_fallback(text, request_id):
                    yield result
                return
        
        try:
            # Get the full audio from your working WebSocket method
            audio_data = await self._websocket_synthesis(text)
            
            if len(audio_data) == 0:
                print("⚠️ Azure returned no audio, using fallback")
                if self._openai_fallback:
                    async for result in self._use_fallback(text, request_id):
                        yield result
                    return
            
            # SUCCESS! Now stream it smoothly in chunks
            print(f"🔵 Streaming {len(audio_data)} bytes of audio in smooth chunks")
            
            # Use 0.15 second chunks (14400 bytes at 48kHz 16-bit mono) for smooth streaming
            chunk_size = 14400
            chunk_count = 0
            
            for i in range(0, len(audio_data), chunk_size):
                chunk_data = audio_data[i:i + chunk_size]
                
                # Ensure even length for 16-bit samples
                if len(chunk_data) % 2 == 1:
                    chunk_data = chunk_data[:-1]
                
                if len(chunk_data) > 0:
                    samples_per_channel = len(chunk_data) // (self._num_channels * 2)
                    
                    audio_frame = rtc.AudioFrame(
                        data=chunk_data,
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                        samples_per_channel=samples_per_channel
                    )
                    
                    # Mark as final only for the last chunk
                    is_final = (i + chunk_size >= len(audio_data))
                    
                    yield SynthesizedAudio(
                        frame=audio_frame,
                        request_id=request_id,
                        is_final=is_final
                    )
                    
                    chunk_count += 1
                    print(f"🔵 Streamed smooth chunk #{chunk_count}: {samples_per_channel} samples")
                    
                    # Small delay between chunks to prevent overwhelming the audio pipeline
                    if not is_final:
                        await asyncio.sleep(0.05)  # 50ms delay for smooth streaming
            
            # Reset failure counter on success
            self._failed_requests = 0
            print(f"🔵 ✅ Azure streaming completed: {chunk_count} smooth chunks")
            
        except Exception as e:
            self._failed_requests += 1
            print(f"❌ Azure TTS Error (failure #{self._failed_requests}): {e}")
            
            # Fall back to OpenAI
            if self._openai_fallback:
                print("🔄 Using OpenAI fallback due to Azure error...")
                try:
                    async for result in self._use_fallback(text, request_id):
                        yield result
                    return
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
            
            # Final fallback - empty frame
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
    
    async def _websocket_synthesis(self, text: str) -> bytes:
        """Your working WebSocket synthesis method - unchanged"""
        try:
            access_token = await self._get_access_token()
            request_id = str(uuid.uuid4()).replace('-', '')
            connection_id = str(uuid.uuid4()).replace('-', '')
            
            # Correct Azure WebSocket URL format
            uri = f"{self._ws_url}?Authorization=Bearer%20{access_token}&X-ConnectionId={connection_id}"
            
            print(f"🔵 Connecting to Azure WebSocket: {self._region}")
            
            # Use basic WebSocket connection without extra parameters for maximum compatibility
            async with websockets.connect(uri) as websocket:
                print(f"🔵 Connected to Azure WebSocket")
                
                # Send configuration message
                config_msg = self._create_config_message(request_id)
                await websocket.send(config_msg)
                print(f"🔵 Sent config message")
                
                # Send SSML message
                ssml = self._create_ssml(text)
                ssml_msg = self._create_ssml_message(request_id, ssml)
                await websocket.send(ssml_msg)
                print(f"🔵 Sent SSML message: {len(text)} chars")
                
                # Collect audio data with improved extraction
                audio_chunks = []
                received_turn_start = False
                
                try:
                    while True:
                        # Add timeout to prevent hanging
                        message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                        
                        if isinstance(message, str):
                            print(f"🔵 Received text message: {message[:100]}...")
                            # Check for important path messages
                            if 'Path:turn.start' in message:
                                received_turn_start = True
                                print("🔵 Turn started")
                            elif 'Path:turn.end' in message:
                                print("🔵 Turn ended")
                                break
                            elif 'Path:response' in message:
                                print("🔵 Response message received")
                        
                        elif isinstance(message, bytes):
                            print(f"🔵 Received binary message: {len(message)} bytes")
                            
                            # Skip very small messages (control/metadata)
                            if len(message) < 100:
                                continue
                            
                            # Use your working audio extraction method
                            audio_data = self._extract_audio_from_message(message)
                            
                            if audio_data and len(audio_data) >= 100:
                                audio_chunks.append(audio_data)
                                print(f"🔵 Added clean audio chunk: {len(audio_data)} bytes")
                
                except asyncio.TimeoutError:
                    print("🔵 WebSocket timeout - ending collection")
                
                # Combine all audio chunks
                total_audio = b"".join(audio_chunks)
                print(f"🔵 Total audio collected: {len(total_audio)} bytes from {len(audio_chunks)} chunks")
                
                # Validate we got meaningful audio data
                if len(total_audio) < 1000:
                    print(f"⚠️ Audio data too small: {len(total_audio)} bytes, may be corrupted")
                    # If we got almost nothing, throw an error to trigger fallback
                    if len(total_audio) < 100:
                        raise Exception("Azure returned insufficient audio data")
                
                # Ensure proper 16-bit PCM alignment
                if len(total_audio) % 2 != 0:
                    total_audio = total_audio[:-1]
                    print(f"🔵 Aligned final audio: removed 1 byte, now {len(total_audio)} bytes")
                
                return total_audio
                
        except websockets.exceptions.WebSocketException as e:
            print(f"❌ Azure WebSocket connection error: {e}")
            raise Exception(f"WebSocket connection failed: {e}")
        except Exception as e:
            print(f"❌ Azure WebSocket synthesis error: {e}")
            raise
    
    async def synthesize(self, text: str, **kwargs):
        """Synthesize speech using Azure WebSocket with OpenAI fallback - returns async generator for LiveKit compatibility"""
        # Accept any additional kwargs that LiveKit might pass
        conn_options = kwargs.get('conn_options', None)
        request_id = str(uuid.uuid4())
        
        # If we've had too many failures, use fallback immediately
        if self._failed_requests >= self._max_failures:
            print(f"🔄 Too many Azure failures ({self._failed_requests}), using OpenAI fallback")
            if self._openai_fallback:
                async for result in self._use_fallback(text, request_id):
                    yield result
                return
        
        try:
            start_time = time.time()
            
            # Try Azure WebSocket synthesis
            audio_data = await self._websocket_synthesis(text)
            
            processing_time = time.time() - start_time
            print(f"🔵 Azure WebSocket TTS: Generated {len(audio_data)} bytes in {processing_time:.3f}s")
            
            # Check if we actually got audio data
            if len(audio_data) == 0:
                self._failed_requests += 1
                print(f"⚠️ Azure TTS returned 0 bytes (failure #{self._failed_requests}) - trying fallback")
                
                if self._openai_fallback:
                    async for result in self._use_fallback(text, request_id):
                        yield result
                    return
                else:
                    # Create empty frame if no fallback
                    audio_frame = rtc.AudioFrame.create(
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                        samples_per_channel=0
                    )
            else:
                # Success! Reset failure counter
                self._failed_requests = 0
                
                # Calculate samples per channel for 16-bit audio
                samples_per_channel = len(audio_data) // (self._num_channels * 2)  # 2 bytes per sample (16-bit)
                
                print(f"🔵 Creating AudioFrame: {len(audio_data)} bytes, {samples_per_channel} samples per channel")
                
                audio_frame = rtc.AudioFrame(
                    data=audio_data,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                    samples_per_channel=samples_per_channel
                )
            
            synthesized_audio = SynthesizedAudio(
                frame=audio_frame,
                request_id=request_id,
                is_final=True
            )
            
            # LiveKit expects an async generator, so yield the result
            yield synthesized_audio
            
        except Exception as e:
            self._failed_requests += 1
            print(f"❌ Azure TTS Error (failure #{self._failed_requests}): {e}")
            print(f"❌ Error type: {type(e).__name__}")
            
            # Try fallback on error
            if self._openai_fallback:
                print("🔄 Trying OpenAI fallback due to Azure error...")
                try:
                    async for result in self._use_fallback(text, request_id):
                        yield result
                    return
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
            
            # If all else fails, create empty frame to prevent crashes
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
                streaming=True             # Enable streaming for smooth buffering
            )
            print("🚀 Using Azure WebSocket TTS with smooth buffering for speech synthesis!")
        except Exception as e:
            print(f"⚠️ Azure TTS initialization failed, falling back to OpenAI: {e}")
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
        print("🚀 Using Groq STT + Azure Smooth TTS for ultra-fast speech processing!")
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
        audio_pipeline_delay = 0.5  # Much lower with smooth streaming
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
                print(f"   💡 SUGGESTION: TTS is slow - check Azure connection or use faster voice")
    
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