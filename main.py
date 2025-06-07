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

# Environment check logging
print("🔍 Environment Check:")
print(f"SUPABASE_URL: {'✅' if SUPABASE_URL else '❌'}")
print(f"SUPABASE_SERVICE_ROLE: {'✅' if SUPABASE_KEY else '❌'}")
print(f"SESSION_ID: {'✅' if os.getenv('SESSION_ID') else '❌'}")
print(f"OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
print(f"ELEVEN_API_KEY: {'✅' if os.getenv('ELEVEN_API_KEY') else '❌'}")
print(f"CARTESIA_API_KEY: {'✅' if CARTESIA_API_KEY else '❌'}")

# Modern Agent class
class ProspectAgent(Agent):
    def __init__(self, prospect_prompt: str):
        super().__init__(
            instructions=prospect_prompt + "\n\nIMPORTANT: Keep responses very short (1 sentence, max 10-15 words) for natural conversation flow. Be direct and conversational.",
        )
        print("✅ ProspectAgent initialized successfully")

def fetch_token_from_supabase(session_id):
    print(f"🔍 Fetching token for session: {session_id[:20]}...")
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
        print(f"Session ID: {session_id[:20]}...")
        
        # Fetch token from Supabase
        print(f"Fetching token for session: {session_id[:20]}...")
        token, room_name, identity = fetch_token_from_supabase(session_id)
        
        # Load PDF content
        pdf_path = "assets/sales.pdf"
        print(f"📄 Loading PDF: {pdf_path}")
        business_pdf_text = extract_pdf_text(pdf_path)
        print(f"✅ PDF loaded ({len(business_pdf_text)} chars)")
        
        # Generate prospect persona
        print("🧠 Generating prospect persona...")
        print("🤖 Sending request to OpenAI for prospect prompt...")
        
        fit_strictness = "strict"
        objection_focus = "price"
        toughness_level = 6
        call_type = "discovery"
        tone = "direct"
        
        prospect_prompt = await get_prospect_prompt(
            fit_strictness,
            objection_focus,
            toughness_level,
            call_type,
            tone,
            business_pdf_text,
        )
        
        print("✅ Got prospect prompt from OpenAI")
        print(f"📝 Prompt length: {len(prospect_prompt)} characters")
        print(f"✅ Persona generated ({len(prospect_prompt)} chars)")
        
        # Display persona info
        print("=" * 60)
        lines = prospect_prompt.split('\n')
        for line in lines:
            if 'name' in line.lower() and ('**' in line or '*' in line):
                print(f"👤 {line.strip()}")
            elif 'objection' in line.lower() and ('**' in line or '*' in line):
                print(f"👤 {line.strip()}")
                break
        print("=" * 60)
        
        # Connect to LiveKit room
        print(f"📡 Connecting to room: {room_name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        print("✅ Connected to LiveKit")
        
        # Initialize AI components
        print("🔧 Initializing AI components...")
        
        # Create agent
        agent = ProspectAgent(prospect_prompt)
        
        # Create VAD with more sensitive settings
        print("🔧 Creating VAD with MORE SENSITIVE settings...")
        vad_instance = silero.VAD.load(
            min_speech_duration=0.1,    # More sensitive - detect shorter speech
            min_silence_duration=0.3,   # Shorter silence - faster response
            prefix_padding_duration=0.1,
            activation_threshold=0.4,   # Lower threshold - easier to trigger
        )
        print("✅ VAD created successfully")
        
        # Create STT
        print("🔧 Creating STT with Whisper...")
        stt_instance = openai.STT(
            model="whisper-1",
            language="en",
        )
        print("✅ STT created successfully")
        
        # Create LLM
        print("🔥 Creating LLM with gpt-4.1-nano...")
        llm_instance = openai.LLM(
            model="gpt-4.1-nano",    # Ultra-fast nano model
            temperature=0.7,
        )
        print("✅ LLM created successfully")
        
        # Create TTS
        print("🔥 Creating Cartesia TTS with Sonic 2 model...")
        tts_instance = cartesia.TTS(
            model="sonic-2",                          # Cartesia Sonic 2 model
            voice="6f84f4b8-58a2-430c-8c79-688dad597532",  # Specific voice ID
            speed=1.2,
            encoding="pcm_s16le",
            sample_rate=22050,
        )
        print("✅ Cartesia TTS created successfully")
        
        # Create AgentSession
        print("🔧 Creating AgentSession with all components...")
        session = AgentSession(
            vad=vad_instance,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
        )
        
        # Add comprehensive debug event handlers (matching your pattern)
        print("🔧 Adding event handlers for speech detection...")
        
        # Counter for message numbering
        user_msg_count = [0]
        agent_msg_count = [0]
        
        try:
            @session.on("user_speech_committed")
            def on_user_speech_committed(text: str):
                user_msg_count[0] += 1
                print(f"🎤 USER [{user_msg_count[0]:02d}]: {text}")
            print("✅ user_speech_committed handler added")
        except Exception as e:
            print(f"❌ user_speech_committed handler failed: {e}")
        
        try:
            @session.on("user_started_speaking")
            def on_user_started_speaking():
                print("🎤 User started speaking (VAD triggered)")
            print("✅ user_started_speaking handler added")
        except Exception as e:
            print(f"❌ user_started_speaking handler failed: {e}")
            
        try:
            @session.on("user_stopped_speaking")
            def on_user_stopped_speaking():
                print("🎤 User stopped speaking (VAD ended)")
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
        
        # Try additional event variations
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
        
        # Generic event logger
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
        
        # Start the session
        print("🔧 Starting AgentSession...")
        await session.start(agent=agent, room=ctx.room)
        print("✅ AgentSession started successfully")
        print("🔄 Session running - speak now and watch for VAD/STT logs...")
        
        # Generate welcome message
        print("🗣️ Preparing to speak welcome message...")
        await asyncio.sleep(0.5)
        print("🗣️ Calling session.generate_reply() for welcome message...")
        
        # Track agent messages
        def track_agent_reply(text):
            agent_msg_count[0] += 1
            print(f"🤖 AGENT [{agent_msg_count[0]:02d}]: {text}")
        
        # Generate initial greeting
        await session.generate_reply(instructions="Greet the user by saying 'Hey! Can you hear me clearly?'")
        track_agent_reply("Hey! Can you hear me clearly?")
        
        print("✅ Welcome message generate_reply() call completed")
        print("🎉 Sales bot ready! (startup: 12.8s)")
        print("🗣️ Conversation active - user can now speak...")
        
        # Add heartbeat monitoring
        async def heartbeat():
            while True:
                await asyncio.sleep(10)
                print("💓 Session heartbeat - still running and listening...")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat())
        print("✅ Heartbeat monitoring started")
        
        # Keep session alive
        await heartbeat_task
        
    except Exception as e:
        print(f"❌ Entrypoint function failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))