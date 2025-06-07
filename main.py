# main.py - UPDATED WITH MODERN LIVEKIT API + DEBUG LOGGING
import asyncio
import os
import requests
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession, 
    AutoSubscribe, 
    JobContext, 
    WorkerOptions, 
    cli, 
    llm,
    RunContext
)
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero, cartesia, elevenlabs, cartesia
from pdf_utils import extract_pdf_text
from gpt_utils import get_prospect_prompt

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

# DEBUG: Log environment variables
print("🔍 DEBUG - Environment variables in Fly machine:")
print(f"SUPABASE_URL: {'✅ Set' if SUPABASE_URL else '❌ Missing'} - {SUPABASE_URL}")
print(f"SUPABASE_SERVICE_ROLE: {'✅ Set' if SUPABASE_KEY else '❌ Missing'} - {SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")
print(f"SESSION_ID: {'✅ Set' if os.getenv('SESSION_ID') else '❌ Missing'}")
print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
print(f"ELEVEN_API_KEY: {'✅ Set' if os.getenv('ELEVEN_API_KEY') else '❌ Missing'}")
print(f"CARTESIA_API_KEY: {'✅ Set' if CARTESIA_API_KEY else '❌ Missing'}")

# Modern Agent class - REMOVED end_call function to prevent auto-hangups
class ProspectAgent(Agent):
    def __init__(self, prospect_prompt: str):
        super().__init__(
            instructions=prospect_prompt + "\n\nIMPORTANT: Never end the call unless explicitly asked. Stay in character and continue the conversation.",
        )
        print("✅ DEBUG - ProspectAgent created without auto-end function")

def fetch_token_from_supabase(session_id):
    print("🔍 DEBUG - Starting Supabase token fetch...")
    url = f"{SUPABASE_URL}/rest/v1/livekit_tokens?token=eq.{session_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    
    print(f"🔍 DEBUG - Making request to: {url}")
    print(f"🔍 DEBUG - Headers: apikey={SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")
    
    try:
        res = requests.get(url, headers=headers)
        print(f"🔍 DEBUG - Response status: {res.status_code}")
        print(f"🔍 DEBUG - Response text: {res.text[:200]}...")
        
        res.raise_for_status()
        data = res.json()
        if not data:
            print("❌ ERROR - Token not found for session_id")
            raise ValueError("❌ Token not found for session_id")
        
        token_data = data[0]
        print(f"✅ SUCCESS - Retrieved token for room: {token_data['room']}, identity: {token_data['identity']}")
        return token_data['token'], token_data['room'], token_data['identity']
    except Exception as e:
        print(f"❌ ERROR - Supabase fetch failed: {e}")
        raise

async def entrypoint(ctx):
    print("🚀 DEBUG - Starting entrypoint function...")
    
    try:
        session_id = os.getenv("SESSION_ID")
        print(f"🔍 DEBUG - Retrieved session ID: {session_id}")
        
        print("🔍 DEBUG - Fetching token from Supabase...")
        token, room_name, identity = fetch_token_from_supabase(session_id)
        print(f"✅ DEBUG - Token fetch successful for room: {room_name}")
        
        pdf_path = "assets/sales.pdf"
        print(f"📄 DEBUG - Starting PDF extraction: {pdf_path}")
        try:
            business_pdf_text = extract_pdf_text(pdf_path)
            print(f"✅ DEBUG - PDF extraction completed. Text length: {len(business_pdf_text)} characters")
        except Exception as e:
            print(f"❌ ERROR - PDF extraction failed: {e}")
            raise
        
        fit_strictness = "strict"
        objection_focus = "trust"
        toughness_level = 5
        call_type = "discovery"
        tone = "direct"
        
        print("💬 DEBUG - Starting GPT prospect prompt generation...")
        try:
            prospect_prompt = await get_prospect_prompt(
                fit_strictness,
                objection_focus,
                toughness_level,
                call_type,
                tone,
                business_pdf_text,
            )
            print(f"✅ DEBUG - GPT prospect prompt generated. Length: {len(prospect_prompt)} characters")
        except Exception as e:
            print(f"❌ ERROR - GPT prompt generation failed: {e}")
            raise
        
        print("\n🧠 GPT Persona Prompt:\n")
        print(prospect_prompt)
        print("\n" + "="*60 + "\n")
        
        print(f"📡 DEBUG - Attempting to connect to LiveKit room: '{room_name}'")
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            print("✅ DEBUG - Successfully connected to LiveKit room")
        except Exception as e:
            print(f"❌ ERROR - LiveKit connection failed: {e}")
            raise
        
        print("🔧 DEBUG - Creating Agent and AgentSession with gpt-4.1-nano + Cartesia Sonic 2...")
        
        try:
            print("🔧 DEBUG - Creating ProspectAgent with instructions...")
            agent = ProspectAgent(prospect_prompt)
            print("✅ DEBUG - ProspectAgent created successfully")
            
            print("🔧 DEBUG - Creating VAD with MORE SENSITIVE settings...")
            vad_instance = silero.VAD.load(
                min_speech_duration=0.1,    # 🔥 MORE SENSITIVE - detect shorter speech
                min_silence_duration=0.3,   # 🔥 SHORTER SILENCE - faster response
                prefix_padding_duration=0.1,
                activation_threshold=0.4,   # 🔥 LOWER THRESHOLD - easier to trigger
            )
            print("✅ DEBUG - VAD created successfully")
            
            print("🔧 DEBUG - Creating STT with Whisper...")
            stt_instance = openai.STT(
                model="whisper-1",
                language="en",
            )
            print("✅ DEBUG - STT created successfully")
            
            print("🔥 DEBUG - Creating LLM with gpt-4.1-nano...")
            llm_instance = openai.LLM(
                model="gpt-4.1-nano",    # 🔥 ULTRA-FAST NANO MODEL
                temperature=0.7,
                # max_tokens handled by the model internally
            )
            print("✅ DEBUG - LLM created successfully")
            
            print("🔥 DEBUG - Creating Cartesia TTS with Sonic 2 model...")
            tts_instance = cartesia.TTS(
                model="sonic-2",                          # 🔥 CARTESIA SONIC 2 MODEL
                voice="6f84f4b8-58a2-430c-8c79-688dad597532",  # 🔥 SPECIFIC VOICE ID
                speed=1.0,
                encoding="pcm_s16le",
                sample_rate=24000,
            )
            print("✅ DEBUG - Cartesia TTS created successfully")
            
            print("🔧 DEBUG - Creating AgentSession with all components...")
            session = AgentSession(
                vad=vad_instance,
                stt=stt_instance,
                llm=llm_instance,
                tts=tts_instance,
            )
            
            # 🔍 ADD DEBUG EVENT HANDLERS
            print("🔧 DEBUG - Adding event handlers for speech detection...")
            
            # Add MORE event handlers to catch everything
            try:
                @session.on("user_speech_committed")
                def on_user_speech_committed(text: str):
                    print(f"🎤 DEBUG - User speech committed: '{text}' (length: {len(text)})")
                print("✅ DEBUG - user_speech_committed handler added")
            except Exception as e:
                print(f"❌ DEBUG - user_speech_committed handler failed: {e}")
            
            try:
                @session.on("user_started_speaking")
                def on_user_started_speaking():
                    print("🎤 DEBUG - User started speaking (VAD triggered)")
                print("✅ DEBUG - user_started_speaking handler added")
            except Exception as e:
                print(f"❌ DEBUG - user_started_speaking handler failed: {e}")
                
            try:
                @session.on("user_stopped_speaking")
                def on_user_stopped_speaking():
                    print("🎤 DEBUG - User stopped speaking (VAD ended)")
                print("✅ DEBUG - user_stopped_speaking handler added")
            except Exception as e:
                print(f"❌ DEBUG - user_stopped_speaking handler failed: {e}")
            
            try:
                @session.on("agent_started_speaking")
                def on_agent_started_speaking():
                    print("🗣️ DEBUG - Agent started speaking")
                print("✅ DEBUG - agent_started_speaking handler added")
            except Exception as e:
                print(f"❌ DEBUG - agent_started_speaking handler failed: {e}")
                
            try:
                @session.on("agent_stopped_speaking") 
                def on_agent_stopped_speaking():
                    print("🗣️ DEBUG - Agent stopped speaking")
                print("✅ DEBUG - agent_stopped_speaking handler added")
            except Exception as e:
                print(f"❌ DEBUG - agent_stopped_speaking handler failed: {e}")
            
            # Try alternative event names in case the above don't work
            try:
                @session.on("speech_recognized")
                def on_speech_recognized(text: str):
                    print(f"🎤 DEBUG - Speech recognized: '{text}'")
                print("✅ DEBUG - speech_recognized handler added")
            except Exception as e:
                print(f"❌ DEBUG - speech_recognized handler failed: {e}")
                
            try:
                @session.on("user_transcript")
                def on_user_transcript(text: str):
                    print(f"🎤 DEBUG - User transcript: '{text}'")
                print("✅ DEBUG - user_transcript handler added")
            except Exception as e:
                print(f"❌ DEBUG - user_transcript handler failed: {e}")
            
            # Try even more event variations
            try:
                @session.on("stt_final_transcript")
                def on_stt_final_transcript(text: str):
                    print(f"🎤 DEBUG - STT final transcript: '{text}'")
                print("✅ DEBUG - stt_final_transcript handler added")
            except Exception as e:
                print(f"❌ DEBUG - stt_final_transcript handler failed: {e}")
                
            try:
                @session.on("vad_speech_start")
                def on_vad_speech_start():
                    print("🎤 DEBUG - VAD speech start detected")
                print("✅ DEBUG - vad_speech_start handler added")
            except Exception as e:
                print(f"❌ DEBUG - vad_speech_start handler failed: {e}")
                
            try:
                @session.on("vad_speech_end")
                def on_vad_speech_end():
                    print("🎤 DEBUG - VAD speech end detected")
                print("✅ DEBUG - vad_speech_end handler added")
            except Exception as e:
                print(f"❌ DEBUG - vad_speech_end handler failed: {e}")
            
            # Add a generic catch-all event listener
            try:
                original_emit = session.emit
                def debug_emit(event, *args, **kwargs):
                    print(f"🔄 DEBUG - Event emitted: '{event}' with args: {args}")
                    return original_emit(event, *args, **kwargs)
                session.emit = debug_emit
                print("✅ DEBUG - Generic event logger added")
            except Exception as e:
                print(f"❌ DEBUG - Generic event logger failed: {e}")
            
            print("✅ DEBUG - AgentSession created successfully")
            
        except Exception as e:
            print(f"❌ ERROR - Agent/Session setup failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print("🔧 DEBUG - Starting AgentSession...")
            await session.start(agent=agent, room=ctx.room)
            print("✅ DEBUG - AgentSession started successfully")
            print("🔄 DEBUG - Session running - speak now and watch for VAD/STT logs...")
            
            # Add a heartbeat to confirm the session is active
            async def heartbeat():
                while True:
                    await asyncio.sleep(10)
                    print("💓 DEBUG - Session heartbeat - still running and listening...")
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(heartbeat())
            print("✅ DEBUG - Heartbeat monitoring started")
        except Exception as e:
            print(f"❌ ERROR - AgentSession start failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print("🗣️ DEBUG - Preparing to speak welcome message...")
            await asyncio.sleep(0.5)
            print("🗣️ DEBUG - Calling session.generate_reply() for welcome message...")
            
            await session.generate_reply(instructions="Greet the user by saying 'Hey! Can you hear me clearly?'")
            
            print("✅ DEBUG - Welcome message generate_reply() call completed")
        except Exception as e:
            print(f"❌ ERROR - Welcome message failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ ERROR - Entrypoint function failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))