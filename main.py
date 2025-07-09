"""
LiveKit AI Sales Bot with Azure Speech Services TTS Integration (HYBRID WEBSOCKET + REST)

SETUP INSTRUCTIONS:
1. Set environment variables:
   - AZURE_SPEECH_API_KEY=your_azure_speech_api_key
   - AZURE_SPEECH_REGION=your_region (e.g., eastus, westus2, etc.)
   
2. Get Azure Speech credentials:
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


# Azure TTS using REST API as primary with WebSocket fallback + Audio Pipelining
class AzureHybridTTS(TTS):
    """Azure Speech Services TTS with REST API, WebSocket fallback, and audio pipelining"""
    
    def __init__(
        self,
        api_key: str,
        region: str,
        voice: str = "en-US-AriaNeural",
        speed: float = 1.0,
        streaming: bool = True  # Enable streaming for pipelining
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
        self._failed_requests = 0
        self._max_failures = 3
        self._use_websocket = False  # Start with REST API
        
        # Azure endpoints
        self._token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
        self._rest_url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self._ws_url = f"wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
        
        # Pipelining state
        self._synthesis_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
        self._synthesis_worker_task = None
        self._is_streaming = False
        
        # Create OpenAI fallback
        self._openai_fallback = None
        try:
            self._openai_fallback = lk_openai.TTS(voice="alloy", speed=speed)
            print("🔄 OpenAI TTS fallback initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI fallback: {e}")
        
        print(f"🔵 Azure Hybrid TTS initialized with voice: {voice}, speed: {speed}, pipelining: enabled")
    
    async def _start_synthesis_worker(self):
        """Background worker that continuously processes synthesis requests"""
        print("🎵 Starting audio synthesis pipeline worker")
        
        while self._is_streaming:
            try:
                # Wait for next synthesis request
                synthesis_request = await asyncio.wait_for(
                    self._synthesis_queue.get(), 
                    timeout=1.0
                )
                
                if synthesis_request is None:  # Shutdown signal
                    break
                
                text_chunk, request_id, chunk_index = synthesis_request
                
                print(f"🎵 Pipeline: Synthesizing chunk #{chunk_index}: '{text_chunk[:30]}...'")
                
                # Synthesize this chunk
                try:
                    async for audio_result in self._synthesize_hybrid(text_chunk, f"{request_id}-{chunk_index}"):
                        # Add synthesized audio to the audio queue with chunk info
                        await self._audio_queue.put((audio_result, chunk_index, False))  # False = not final
                        break  # We expect one result per chunk
                        
                except Exception as e:
                    print(f"❌ Pipeline synthesis error for chunk #{chunk_index}: {e}")
                    # Put an error marker in the queue
                    await self._audio_queue.put((None, chunk_index, False))
                
                print(f"✅ Pipeline: Chunk #{chunk_index} synthesis complete")
                
            except asyncio.TimeoutError:
                # No new requests, continue waiting
                continue
            except Exception as e:
                print(f"❌ Synthesis worker error: {e}")
                await asyncio.sleep(0.1)
        
        print("🎵 Synthesis pipeline worker stopped")
    
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
        if self.capabilities.streaming:
            return self._PipelinedStreamingContext(self)
        else:
            return self._SimpleStreamingContext(self)
    
    class _SimpleStreamingContext:
        """Simple async context manager for LiveKit streaming - no pipelining"""
        
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
                self._audio_generator = self.tts._synthesize_hybrid(text, request_id)
            
            try:
                # Get next audio chunk
                audio_chunk = await self._audio_generator.__anext__()
                return audio_chunk
            except StopAsyncIteration:
                # Done with this text, reset for next
                self._audio_generator = None
                raise StopAsyncIteration
            except Exception as e:
                print(f"❌ Azure TTS error: {e}")
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
    
    class _PipelinedStreamingContext:
        """Async context manager for LiveKit streaming with audio pipelining"""
        
        def __init__(self, tts_instance):
            self.tts = tts_instance
            self._current_text = None
            self._audio_generator = None
            self._request_id = None
            self._chunk_counter = 0
        
        async def __aenter__(self):
            """Enter the async context manager and start pipelining"""
            self.tts._is_streaming = True
            
            # Start the background synthesis worker
            self.tts._synthesis_worker_task = asyncio.create_task(
                self.tts._start_synthesis_worker()
            )
            
            print("🎵 Audio pipelining enabled")
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Exit the async context manager and cleanup pipelining"""
            self.tts._is_streaming = False
            
            # Signal worker to stop
            try:
                await self.tts._synthesis_queue.put(None)  # Shutdown signal
                
                if self.tts._synthesis_worker_task:
                    await asyncio.wait_for(self.tts._synthesis_worker_task, timeout=2.0)
            except asyncio.TimeoutError:
                print("⚠️ Synthesis worker didn't stop cleanly")
                if self.tts._synthesis_worker_task:
                    self.tts._synthesis_worker_task.cancel()
            
            # Clear queues
            while not self.tts._synthesis_queue.empty():
                try:
                    self.tts._synthesis_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            while not self.tts._audio_queue.empty():
                try:
                    self.tts._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            print("🎵 Audio pipelining disabled")
        
        def __aiter__(self):
            """Make this object async iterable"""
            return self
        
        async def __anext__(self):
            """Async iterator protocol - return pipelined audio chunks"""
            if self._audio_generator is None:
                # Wait for text to be sent via send()
                while self._current_text is None:
                    await asyncio.sleep(0.01)
                
                # Process the text for pipelining
                full_text = self._current_text
                self._current_text = None
                self._request_id = str(uuid.uuid4())
                
                # Split text into chunks for pipelining (simple sentence-based splitting)
                text_chunks = self._split_text_for_pipelining(full_text)
                
                print(f"🎵 Pipelining {len(text_chunks)} text chunks")
                
                # Queue all chunks for synthesis (they'll be processed in parallel)
                for i, chunk in enumerate(text_chunks):
                    await self.tts._synthesis_queue.put((chunk, self._request_id, i))
                
                # Create audio generator that yields results as they're ready
                self._audio_generator = self._get_pipelined_audio(len(text_chunks))
            
            try:
                # Get next audio chunk from the pipeline
                audio_chunk = await self._audio_generator.__anext__()
                return audio_chunk
            except StopAsyncIteration:
                # Done with this text, reset for next
                self._audio_generator = None
                self._chunk_counter = 0
                raise StopAsyncIteration
            except Exception as e:
                print(f"❌ Pipelined audio error: {e}")
                # Reset and signal completion
                self._audio_generator = None
                raise StopAsyncIteration
        
        def _split_text_for_pipelining(self, text: str) -> List[str]:
            """Split text into chunks suitable for pipelining"""
            # Simple sentence-based splitting
            import re
            
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            
            # Group sentences into chunks (target ~50-100 chars per chunk for good pipelining)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > 100 and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Ensure we have at least one chunk
            if not chunks:
                chunks = [text]
            
            return chunks
        
        async def _get_pipelined_audio(self, expected_chunks: int):
            """Generator that yields audio chunks as they become available"""
            chunks_received = 0
            next_chunk_to_yield = 0
            chunk_buffer = {}  # Buffer to store out-of-order chunks
            
            while chunks_received < expected_chunks:
                try:
                    # Wait for next audio chunk
                    audio_result, chunk_index, is_final = await asyncio.wait_for(
                        self.tts._audio_queue.get(),
                        timeout=10.0
                    )
                    
                    chunks_received += 1
                    
                    if audio_result is None:
                        print(f"⚠️ Chunk #{chunk_index} failed synthesis")
                        continue
                    
                    # Store the chunk
                    chunk_buffer[chunk_index] = audio_result
                    
                    # Yield chunks in order
                    while next_chunk_to_yield in chunk_buffer:
                        chunk_audio = chunk_buffer.pop(next_chunk_to_yield)
                        
                        # Mark as final if this is the last chunk
                        final_chunk = (next_chunk_to_yield == expected_chunks - 1)
                        
                        yield SynthesizedAudio(
                            frame=chunk_audio.frame,
                            request_id=chunk_audio.request_id,
                            is_final=final_chunk
                        )
                        
                        print(f"🎵 ✅ Yielded pipelined chunk #{next_chunk_to_yield}")
                        next_chunk_to_yield += 1
                
                except asyncio.TimeoutError:
                    print(f"⚠️ Timeout waiting for audio chunk {chunks_received}/{expected_chunks}")
                    break
                except Exception as e:
                    print(f"❌ Error getting pipelined audio: {e}")
                    break
            
            print(f"🎵 Pipelining complete: {chunks_received}/{expected_chunks} chunks processed")
        
        async def send(self, text: str):
            """Send text to be synthesized with pipelining"""
            print(f"🎵 Received text for pipelined synthesis: '{text[:50]}...'")
            self._current_text = text
    
    async def _synthesize_rest(self, text: str, request_id: str):
        """Primary method: Azure REST API synthesis"""
        try:
            access_token = await self._get_access_token()
            ssml = self._create_ssml(text)
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/ssml+xml',
                'X-Microsoft-OutputFormat': 'raw-48khz-16bit-mono-pcm',
                'User-Agent': 'LiveKit-Azure-TTS'
            }
            
            print(f"🔵 Azure REST TTS: '{text[:50]}...'")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._rest_url, 
                    headers=headers, 
                    data=ssml.encode('utf-8'),
                    timeout=30
                ) as response:
                    
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        if len(audio_data) >= 2:
                            # Ensure proper 16-bit alignment
                            if len(audio_data) % 2 == 1:
                                audio_data = audio_data[:-1]
                            
                            samples_per_channel = len(audio_data) // 2
                            
                            frame = rtc.AudioFrame(
                                data=audio_data,
                                sample_rate=self._sample_rate,
                                num_channels=self._num_channels,
                                samples_per_channel=samples_per_channel
                            )
                            
                            print(f"🔵 ✅ REST synthesis complete: {samples_per_channel} samples ({len(audio_data)} bytes)")
                            
                            yield SynthesizedAudio(
                                frame=frame,
                                request_id=request_id,
                                is_final=True
                            )
                            return
                        else:
                            print("🔵 ❌ Empty audio response from Azure REST")
                    else:
                        print(f"🔵 ❌ Azure REST API error: {response.status}")
                        error_text = await response.text()
                        print(f"🔵 Error details: {error_text}")
                        raise Exception(f"REST API failed with status {response.status}")
            
        except Exception as e:
            print(f"🔵 REST synthesis failed: {e}")
            raise e
    
    async def _synthesize_websocket(self, text: str, request_id: str):
        """Fallback method: Azure WebSocket synthesis with improved audio extraction"""
        try:
            access_token = await self._get_access_token()
            connection_id = str(uuid.uuid4()).replace('-', '')
            uri = f"{self._ws_url}?Authorization=Bearer%20{access_token}&X-ConnectionId={connection_id}"
            
            print(f"🔵 Azure WebSocket TTS: '{text[:50]}...'")
            
            async with websockets.connect(uri) as websocket:
                # Send config
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
                
                config_msg = f"X-RequestId:{request_id}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n{json.dumps(config)}"
                await websocket.send(config_msg)
                
                # Send SSML
                ssml = self._create_ssml(text)
                ssml_msg = f"X-RequestId:{request_id}\r\nContent-Type:application/ssml+xml\r\nPath:ssml\r\n\r\n{ssml}"
                await websocket.send(ssml_msg)
                
                # Collect audio with improved extraction
                audio_chunks = []
                
                try:
                    while True:
                        message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                        
                        if isinstance(message, str):
                            if 'Path:turn.end' in message:
                                break
                        elif isinstance(message, bytes):
                            audio_data = self._extract_audio_carefully(message)
                            if audio_data and len(audio_data) >= 1000:
                                audio_chunks.append(audio_data)
                                print(f"🔵 WebSocket chunk: {len(audio_data)} bytes")
                
                except asyncio.TimeoutError:
                    print(f"🔵 WebSocket timeout")
                
                if audio_chunks:
                    combined_audio = b"".join(audio_chunks)
                    
                    if len(combined_audio) % 2 == 1:
                        combined_audio = combined_audio[:-1]
                    
                    if len(combined_audio) >= 2:
                        samples_per_channel = len(combined_audio) // 2
                        
                        frame = rtc.AudioFrame(
                            data=combined_audio,
                            sample_rate=self._sample_rate,
                            num_channels=self._num_channels,
                            samples_per_channel=samples_per_channel
                        )
                        
                        print(f"🔵 ✅ WebSocket synthesis complete: {samples_per_channel} samples")
                        
                        yield SynthesizedAudio(
                            frame=frame,
                            request_id=request_id,
                            is_final=True
                        )
                        return
                
                raise Exception("No audio received from WebSocket")
            
        except Exception as e:
            print(f"🔵 WebSocket synthesis failed: {e}")
            raise e
    
    def _extract_audio_carefully(self, message: bytes) -> Optional[bytes]:
        """Carefully extract audio from WebSocket message"""
        try:
            if b'Path:audio' not in message:
                return None
            
            # Find the header boundary more carefully
            boundary_patterns = [b'\r\n\r\n', b'\n\n']
            audio_start = None
            
            for pattern in boundary_patterns:
                pos = message.find(pattern)
                if pos != -1:
                    audio_start = pos + len(pattern)
                    break
            
            if audio_start is None:
                return None
            
            # Extract raw audio
            raw_audio = message[audio_start:]
            
            # Skip if too small
            if len(raw_audio) < 1000:
                return None
            
            # Ensure alignment
            if len(raw_audio) % 2 == 1:
                raw_audio = raw_audio[:-1]
            
            return raw_audio
            
        except Exception as e:
            print(f"🔵 Audio extraction error: {e}")
            return None
    
    async def _synthesize_hybrid(self, text: str, request_id: str):
        """Hybrid synthesis: Try REST first, then WebSocket, then OpenAI"""
        if self._failed_requests >= self._max_failures:
            print(f"🔄 Too many Azure failures ({self._failed_requests}), using OpenAI fallback")
            if self._openai_fallback:
                async for result in self._use_fallback(text, request_id):
                    yield result
                return
        
        # Try REST API first (most reliable)
        if not self._use_websocket:
            try:
                async for result in self._synthesize_rest(text, request_id):
                    # Success with REST
                    self._failed_requests = 0
                    yield result
                    return
            except Exception as e:
                print(f"🔵 REST failed, trying WebSocket: {e}")
                self._use_websocket = True  # Switch to WebSocket for future requests
        
        # Try WebSocket as fallback
        try:
            async for result in self._synthesize_websocket(text, request_id):
                # Success with WebSocket
                self._failed_requests = 0
                yield result
                return
        except Exception as e:
            print(f"🔵 WebSocket also failed: {e}")
            self._failed_requests += 1
        
        # Final fallback to OpenAI
        if self._openai_fallback:
            print("🔄 Using OpenAI fallback...")
            try:
                async for result in self._use_fallback(text, request_id):
                    yield result
                return
            except Exception:
                pass
        
        # Absolute final fallback
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
        async for result in self._synthesize_hybrid(text, request_id):
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

IMPORTANT: You may hear quick acknowledgments like "Got it!" or "Absolutely!" before your responses. These are system-generated to reduce response delay. Simply continue with your natural response as if you're having a smooth conversation.

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
            tts_engine = AzureHybridTTS(
                api_key=azure_api_key,
                region=azure_region,
                voice="en-US-AriaNeural",  # Professional female voice
                speed=1.1,                 # 10% faster speech
                streaming=False            # Keep simple for now, enable pipelining later
            )
            print("🚀 Using Azure Hybrid TTS (REST + WebSocket) for reliable audio!")
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
        print("🚀 Using Groq STT + Azure Hybrid TTS for ultra-fast, reliable audio!")
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
        audio_pipeline_delay = 0.15  # Lower with hybrid approach
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
                print(f"   💡 SUGGESTION: Hybrid TTS should provide reliable synthesis!")
    
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
                
                # 🚀 IMMEDIATE RESPONSE: Send quick acknowledgment to start TTS right away
                print("⚡ Sending immediate acknowledgment to reduce delay...")
                asyncio.create_task(send_immediate_acknowledgment())
                
            elif event.item.role == 'assistant':
                timing['llm_complete'] = asyncio.get_event_loop().time()
                print(f"🧠 LLM response ready: {event.item.text_content[:50]}...")
                if timing['llm_start']:
                    llm_time = timing['llm_complete'] - timing['llm_start']
                    print(f"🧠 LLM processing took {llm_time:.3f}s")
                print(f"📝 Bot message added to chat: {event.item.text_content}")
    
    async def send_immediate_acknowledgment():
        """Send a quick acknowledgment immediately to start TTS and reduce perceived delay"""
        try:
            # Wait a tiny bit to ensure STT has fully completed
            await asyncio.sleep(0.1)
            
            # Pick a quick acknowledgment phrase
            acknowledgments = [
                "Absolutely!",
                "Got it!",
                "That's great!",
                "Perfect!",
                "I hear you!",
                "Interesting!",
                "Makes sense!"
            ]
            
            import random
            quick_response = random.choice(acknowledgments)
            
            print(f"⚡ Immediate acknowledgment: '{quick_response}'")
            
            # Send the quick response immediately to get TTS started
            await session.say(quick_response, allow_interruptions=True)
            
        except Exception as e:
            print(f"❌ Error sending immediate acknowledgment: {e}")
            # Don't let this break the main flow
    
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