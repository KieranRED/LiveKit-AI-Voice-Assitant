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
print("🔍 Environment Check:")
print(f"SUPABASE_URL: {'✅' if SUPABASE_URL else '❌'}")
print(f"SUPABASE_SERVICE_ROLE: {'✅' if SUPABASE_KEY else '❌'}")
print(f"SESSION_ID: {'✅' if os.getenv('SESSION_ID') else '❌'}")
print(f"OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
print(f"ELEVEN_API_KEY: {'✅' if os.getenv('ELEVEN_API_KEY') else '❌'}")
print(f"CARTESIA_API_KEY: {'✅' if CARTESIA_API_KEY else '❌'}")

# Modern Agent class - REMOVED end_call function to prevent auto-hangups
class ProspectAgent(Agent):
    def __init__(self, prospect_prompt: str):
        super().__init__(
            instructions=prospect_prompt + "\n\nIMPORTANT: Never end the call unless explicitly asked. Stay in character and continue the conversation.",
        )

def fetch_token_from_supabase(session_id):
    print(f"🔍 Fetching token for session: {session_id}")
    url = f"{SUPABASE_URL}/rest/v1/livekit_tokens?token=eq.{session_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        data = res.json()
        if not data:
            raise ValueError("❌ Token not found for session_id")
        
        token_data = data[0]
        print(f"✅ Token retrieved | Room: {token_data['room']} | Identity: {token_data['identity']}")
        return token_data['token'], token_data['room'], token_data['identity']
    except Exception as e:
        print(f"❌ Supabase fetch failed: {e}")
        raise

async def entrypoint(ctx: JobContext):
    print("🚀 Starting AI Sales Bot...")
    
    try:
        session_id = os.getenv("SESSION_ID")
        print(f"Session ID: {session_id}")
        print(f"Fetching token for session: {session_id}")
        
        token, room_name, identity = fetch_token_from_supabase(session_id)
        
        pdf_path = "assets/sales.pdf"
        print(f"📄 Loading PDF: {pdf_path}")
        business_pdf_text = extract_pdf_text(pdf_path)
        print(f"✅ PDF loaded ({len(business_pdf_text)} chars)")
        
        print("🧠 Generating prospect persona...")
        fit_strictness = "strict"
        objection_focus = "trust"
        toughness_level = 5
        call_type = "discovery"
        tone = "direct"
        
        print("🤖 Sending request to OpenAI for prospect prompt...")
        prospect_prompt = await get_prospect_prompt(
            fit_strictness,
            objection_focus,
            toughness_level,
            call_type,
            tone,
            business_pdf_text,
        )
        print(f"✅ Got prospect prompt from OpenAI")
        print(f"📝 Prompt length: {len(prospect_prompt)} characters")
        print(f"✅ Persona generated ({len(prospect_prompt)} chars)")
        
        # Extract name and business from prompt for display
        lines = prospect_prompt.split('\n')
        name_line = next((line for line in lines if '**Name**' in line or '**Name:**' in line), "Unknown")
        business_line = next((line for line in lines if '**Business' in line), "Unknown Business")
        
        print("=" * 60)
        print(f"👤 {name_line}")
        print(f"👤 {business_line}")
        print("=" * 60)
        
        print(f"📡 Connecting to room: {room_name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        print("✅ Connected to LiveKit")
        
        print("🔧 Initializing AI components...")
        agent = ProspectAgent(prospect_prompt)
        print("✅ ProspectAgent initialized successfully")
        
        print("🔧 Creating VAD with MORE SENSITIVE settings...")
        vad_instance = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            prefix_padding_duration=0.1,
            activation_threshold=0.4,
        )
        print("✅ VAD created successfully")
        
        print("🔧 Creating STT with Whisper...")
        stt_instance = openai.STT(
            model="whisper-1",
            language="en",
        )
        print("✅ STT created successfully")
        
        print("🔥 Creating LLM with gpt-4.1-nano...")
        llm_instance = openai.LLM(
            model="gpt-4.1-nano",
            temperature=0.7,
        )
        print("✅ LLM created successfully")
        
        print("🔥 Creating Cartesia TTS with Sonic 2 model...")
        tts_instance = cartesia.TTS(
            model="sonic-2",
            voice="6f84f4b8-58a2-430c-8c79-688dad597532",
            speed=1.0,
            encoding="pcm_s16le",
            sample_rate=24000,
        )
        print("✅ Cartesia TTS created successfully")
        
        print("🔧 Creating AgentSession with all components...")
        session = AgentSession(
            vad=vad_instance,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
        )
        
        print("🔧 Adding event handlers for speech detection...")
        
        # Track message counts
        agent_msg_count = [0]
        
        try:
            @session.on("user_speech_committed")
            def on_user_speech_committed(text: str):
                print(f"🎤 User speech: '{text}'")
            print("✅ user_speech_committed handler added")
        except Exception as e:
            print(f"❌ user_speech_committed handler failed: {e}")
        
        try:
            @session.on("user_started_speaking")
            def on_user_started_speaking():
                print("🎤 User started speaking")
            print("✅ user_started_speaking handler added")
        except Exception as e:
            print(f"❌ user_started_speaking handler failed: {e}")
            
        try:
            @session.on("user_stopped_speaking")
            def on_user_stopped_speaking():
                print("🎤 User stopped speaking")
            print("✅ user_stopped_speaking handler added")
        except Exception as e:
            print(f"❌ user_stopped_speaking handler failed: {e}")
        
        try:
            @session.on("agent_started_speaking")
            def on_agent_started_speaking():
                print("🗣️ Agent started speaking")
            print("✅ agent_started_speaking handler added")
        except Exception as e:
            print(f"❌ agent_started_speaking handler failed: {e}")
            
        try:
            @session.on("agent_stopped_speaking") 
            def on_agent_stopped_speaking():
                print("🗣️ Agent stopped speaking")
            print("✅ agent_stopped_speaking handler added")
        except Exception as e:
            print(f"❌ agent_stopped_speaking handler failed: {e}")
        
        try:
            @session.on("speech_recognized")
            def on_speech_recognized(text: str):
                print(f"🎤 Speech recognized: '{text}'")
            print("✅ speech_recognized handler added")
        except Exception as e:
            print(f"❌ speech_recognized handler failed: {e}")
            
        try:
            @session.on("user_transcript")
            def on_user_transcript(text: str):
                print(f"🎤 User transcript: '{text}'")
            print("✅ user_transcript handler added")
        except Exception as e:
            print(f"❌ user_transcript handler failed: {e}")
        
        # Add generic event logger
        try:
            original_emit = session.emit
            def debug_emit(event, *args, **kwargs):
                print(f"🔄 Event emitted: '{event}' with args: {args}")
                return original_emit(event, *args, **kwargs)
            session.emit = debug_emit
            print("✅ Generic event logger added")
        except Exception as e:
            print(f"❌ Generic event logger failed: {e}")
        
        print("✅ AgentSession created successfully")
        
        print("🔧 Starting AgentSession...")
        await session.start(agent=agent, room=ctx.room)
        print("✅ AgentSession started successfully")
        print("🔄 Session running - speak now and watch for VAD/STT logs...")
        
        # Generate welcome message
        print("🗣️ Preparing to speak welcome message...")
        await asyncio.sleep(0.5)
        print("🗣️ Calling session.generate_reply() for welcome message...")
        
        await session.generate_reply(instructions="Greet the user by saying 'Hey! Can you hear me clearly?'")
        
        agent_msg_count[0] += 1
        print(f"🤖 AGENT [{agent_msg_count[0]:02d}]: Hey! Can you hear me clearly?")
        print("✅ Welcome message generate_reply() call completed")
        
        print(f"🎉 Sales bot ready! (startup: {12.8}s)")
        print("🗣️ Conversation active - user can now speak...")
        
        # Add heartbeat monitoring
        async def heartbeat():
            while True:
                await asyncio.sleep(10)
                print("💓 Session heartbeat - still running and listening...")
        
        heartbeat_task = asyncio.create_task(heartbeat())
        print("✅ Heartbeat monitoring started")
        
    except Exception as e:
        print(f"❌ Entrypoint failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))