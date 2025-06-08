# main.py - CLEANED UP VERSION WITH MINIMAL LOGGING
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

# Minimal environment check
print("🔍 Environment Check:")
print(f"SUPABASE_URL: {'✅' if SUPABASE_URL else '❌'}")
print(f"SUPABASE_SERVICE_ROLE: {'✅' if SUPABASE_KEY else '❌'}")
print(f"SESSION_ID: {'✅' if os.getenv('SESSION_ID') else '❌'}")
print(f"OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
print(f"ELEVEN_API_KEY: {'✅' if os.getenv('ELEVEN_API_KEY') else '❌'}")
print(f"CARTESIA_API_KEY: {'✅' if CARTESIA_API_KEY else '❌'}")

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
        token, room_name, identity = fetch_token_from_supabase(session_id)
        
        # Load PDF and generate persona
        pdf_path = "assets/sales.pdf"
        print(f"📄 Loading PDF: {pdf_path}")
        business_pdf_text = extract_pdf_text(pdf_path)
        print(f"✅ PDF loaded ({len(business_pdf_text)} chars)")
        
        print("🧠 Generating prospect persona...")
        prospect_prompt = await get_prospect_prompt(
            "strict", "trust", 5, "discovery", "direct", business_pdf_text
        )
        
        # Extract and display persona info
        lines = prospect_prompt.split('\n')
        name_line = next((line for line in lines if '**Name**' in line or '**Name:**' in line), "Unknown")
        business_line = next((line for line in lines if '**Business' in line), "Unknown Business")
        
        print("=" * 60)
        print(f"👤 {name_line}")
        print(f"👤 {business_line}")
        print("=" * 60)
        
        # Connect to room
        print(f"📡 Connecting to room: {room_name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        print("✅ Connected to LiveKit")
        
        # Initialize components
        print("🔧 Initializing AI components...")
        agent = ProspectAgent(prospect_prompt)
        
        vad_instance = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            prefix_padding_duration=0.1,
            activation_threshold=0.4,
        )
        
        stt_instance = openai.STT(model="whisper-1", language="en")
        llm_instance = openai.LLM(model="gpt-4.1-nano", temperature=0.7)
        tts_instance = cartesia.TTS(
            model="sonic-2",
            voice="6f84f4b8-58a2-430c-8c79-688dad597532",
            speed=1.0,
            encoding="pcm_s16le",
            sample_rate=24000,
        )
        
        # Create session
        session = AgentSession(
            vad=vad_instance,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
        )
        
        # Add voice event handlers using the correct event names for LiveKit 1.0.23
        conversation_count = [0]
        
        @session.on("speech_created")
        def on_speech_created(event):
            if hasattr(event, 'source'):
                if event.source == 'generate_reply':
                    # This is bot speech
                    conversation_count[0] += 1
                    print(f"🤖 BOT SPEAKING [{conversation_count[0]:02d}]")
                elif event.source == 'user':
                    # This is user speech  
                    print(f"🎤 USER SPEAKING")
                else:
                    print(f"🔊 SPEECH: {event.source}")
        
        # Try to capture user input events with different possible names
        try:
            @session.on("user_transcript")
            def on_user_transcript(text):
                print(f"🎤 USER SAID: {text}")
        except:
            pass
            
        try:
            @session.on("transcript")
            def on_transcript(event):
                if hasattr(event, 'text') and hasattr(event, 'participant'):
                    if event.participant != 'assistant':
                        print(f"🎤 USER SAID: {event.text}")
        except:
            pass
        
        # Remove the debug logger since we found what we need
        print("🔧 Speech event handlers added")
        
        # Start session
        print("🔧 Starting session...")
        await session.start(agent=agent, room=ctx.room)
        
        # Send welcome message
        print("🗣️ Sending welcome message...")
        await asyncio.sleep(0.5)
        await session.generate_reply(instructions="Greet the user by saying 'Hey! Can you hear me clearly?'")
        
        print("🎉 Sales bot ready! Conversation active...")
        
        # Minimal heartbeat (optional - remove if you don't want this either)
        async def heartbeat():
            while True:
                await asyncio.sleep(30)  # Reduced frequency
                print("💓 Bot running...")
        
        asyncio.create_task(heartbeat())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))