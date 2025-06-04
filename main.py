# main.py
import asyncio
import os
import requests
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, elevenlabs, cartesia  # 🔥 ADDED CARTESIA IMPORT
from api import AssistantFnc
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
        
        print("🧠 DEBUG - Building chat context...")
        try:
            initial_ctx = llm.ChatContext().append(
                role="system",
                text=prospect_prompt,
            )
            print("✅ DEBUG - Chat context built successfully")
        except Exception as e:
            print(f"❌ ERROR - Chat context build failed: {e}")
            raise
        
        print(f"📡 DEBUG - Attempting to connect to LiveKit room: '{room_name}'")
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            print("✅ DEBUG - Successfully connected to LiveKit room")
        except Exception as e:
            print(f"❌ ERROR - LiveKit connection failed: {e}")
            raise
        
        print("🔧 DEBUG - Starting assistant setup with Cartesia TTS...")
        fnc_ctx = AssistantFnc()
        
        try:
            print("🔧 DEBUG - Creating VAD with settings...")
            vad_instance = silero.VAD.load(
                min_speech_duration=0.2,
                min_silence_duration=1.0,
                prefix_padding_duration=0.2,
                activation_threshold=0.7,
            )
            print("✅ DEBUG - VAD created successfully")
            
            print("🔧 DEBUG - Creating STT with Whisper...")
            stt_instance = openai.STT(
                model="whisper-1",
                language="en",
            )
            print("✅ DEBUG - STT created successfully")
            
            print("🔧 DEBUG - Creating LLM with gpt-4.1-nano...")
            llm_instance = openai.LLM(
                model="gpt-4.1-nano",
                temperature=0.7,
                max_tokens=300,
            )
            print("✅ DEBUG - LLM created successfully")
            
            print("🔧 DEBUG - Creating Cartesia TTS with sonic-2 model...")
            tts_instance = cartesia.TTS(
                model="sonic-2",
                voice="6f84f4b8-58a2-430c-8c79-688dad597532",
                speed=1.0,
                encoding="pcm_s16le",
                sample_rate=24000,
            )
            print("✅ DEBUG - Cartesia TTS created successfully")
            
            print("🔧 DEBUG - Creating VoiceAssistant with all components...")
            assistant = VoiceAssistant(
                vad=vad_instance,
                stt=stt_instance,
                llm=llm_instance,
                tts=tts_instance,
                chat_ctx=initial_ctx,
                preemptive_synthesis=True,
                fnc_ctx=fnc_ctx,
            )
            print("✅ DEBUG - VoiceAssistant created successfully")
            
        except Exception as e:
            print(f"❌ ERROR - Assistant setup failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print("🔧 DEBUG - Setting up event listeners...")
            
            # Event tracking variables
            last_user_speech_time = 0
            last_agent_speech_time = 0
            
            @assistant.on("user_speech_committed")
            def on_user_speech(msg):
                nonlocal last_user_speech_time
                current_time = asyncio.get_event_loop().time()
                if current_time - last_user_speech_time > 0.5:
                    print(f"🎤 DEBUG - USER SPEECH COMMITTED: {msg.content}")
                    last_user_speech_time = current_time
                
            @assistant.on("agent_speech_committed") 
            def on_agent_speech(msg):
                nonlocal last_agent_speech_time
                current_time = asyncio.get_event_loop().time()
                if current_time - last_agent_speech_time > 0.5:
                    print(f"🤖 DEBUG - AGENT SPEECH COMMITTED: {msg.content}")
                    last_agent_speech_time = current_time
                
            @assistant.on("user_started_speaking")
            def on_user_start():
                print("🎤 DEBUG - USER STARTED SPEAKING")
                
            @assistant.on("user_stopped_speaking")
            def on_user_stop():
                print("🎤 DEBUG - USER STOPPED SPEAKING")
                
            @assistant.on("agent_started_speaking")
            def on_agent_start():
                print("🤖 DEBUG - AGENT STARTED SPEAKING")
                
            @assistant.on("agent_stopped_speaking")
            def on_agent_stop(): 
                print("🤖 DEBUG - AGENT STOPPED SPEAKING")
                
            print("✅ DEBUG - Event listeners configured successfully")
            
        except Exception as e:
            print(f"❌ ERROR - Event listener setup failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print("🔧 DEBUG - Starting assistant in room...")
            assistant.start(ctx.room)
            print("✅ DEBUG - Assistant started successfully in room")
        except Exception as e:
            print(f"❌ ERROR - Assistant start failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print("🗣️ DEBUG - Preparing to speak welcome message...")
            await asyncio.sleep(0.5)
            print("🗣️ DEBUG - Calling assistant.say() for welcome message...")
            
            await assistant.say("Hey! Can you hear me clearly?", allow_interruptions=True)
            
            print("✅ DEBUG - Welcome message say() call completed")
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