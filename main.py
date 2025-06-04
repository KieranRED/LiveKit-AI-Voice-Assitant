# main.py
import asyncio
import os
import requests
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, elevenlabs
from api import AssistantFnc
from pdf_utils import extract_pdf_text
from gpt_utils import get_prospect_prompt

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")

# DEBUG: Log environment variables
print("🔍 DEBUG - Environment variables in Fly machine:")
print(f"SUPABASE_URL: {'✅ Set' if SUPABASE_URL else '❌ Missing'} - {SUPABASE_URL}")
print(f"SUPABASE_SERVICE_ROLE: {'✅ Set' if SUPABASE_KEY else '❌ Missing'} - {SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")
print(f"SESSION_ID: {'✅ Set' if os.getenv('SESSION_ID') else '❌ Missing'}")
print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
print(f"ELEVEN_API_KEY: {'✅ Set' if os.getenv('ELEVEN_API_KEY') else '❌ Missing'}")

def fetch_token_from_supabase(session_id):
    url = f"{SUPABASE_URL}/rest/v1/livekit_tokens?token=eq.{session_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    
    print(f"🔍 DEBUG - Making request to: {url}")
    print(f"🔍 DEBUG - Headers: apikey={SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")
    
    res = requests.get(url, headers=headers)
    print(f"🔍 DEBUG - Response status: {res.status_code}")
    print(f"🔍 DEBUG - Response text: {res.text[:200]}...")
    
    res.raise_for_status()
    data = res.json()
    if not data:
        raise ValueError("❌ Token not found for session_id")
    return data[0]['token'], data[0]['room'], data[0]['identity']

async def entrypoint(ctx: JobContext):
    print("🚀 Starting entrypoint...")
    session_id = os.getenv("SESSION_ID")
    print(f"🔍 Using session ID: {session_id}")
    
    token, room_name, identity = fetch_token_from_supabase(session_id)
    
    pdf_path = "assets/sales.pdf"
    print(f"📄 Extracting PDF: {pdf_path}")
    business_pdf_text = extract_pdf_text(pdf_path)
    
    fit_strictness = "strict"
    objection_focus = "trust"
    toughness_level = 5
    call_type = "discovery"
    tone = "direct"
    
    print("💬 Getting prospect prompt from GPT...")
    prospect_prompt = await get_prospect_prompt(
        fit_strictness,
        objection_focus,
        toughness_level,
        call_type,
        tone,
        business_pdf_text,
    )
    
    print("\n🧠 GPT Persona Prompt:\n")
    print(prospect_prompt)
    print("\n" + "="*60 + "\n")
    
    print("🧠 Building chat context...")
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=prospect_prompt,
    )
    
    print(f"📡 Connecting to LiveKit room '{room_name}' with token...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print("✅ Connected to room.")
    
    print("🔧 Setting up assistant with ultra-fast settings...")
    fnc_ctx = AssistantFnc()
    
    try:
        assistant = VoiceAssistant(
            # 🆕 ULTRA-FAST VAD SETTINGS - Prioritize speed
            vad=silero.VAD.load(
                min_speech_duration=0.08,       # Very low - quick detection
                min_silence_duration=0.3,       # Much lower - faster cutoff
                prefix_padding_duration=0.05,   # Minimal padding
                activation_threshold=0.4,       # Lower threshold - more sensitive
            ),
            
            # 🆕 OPTIMIZED STT
            stt=openai.STT(
                model="whisper-1",
                language="en",
            ),
            
            # 🆕 SPEED-OPTIMIZED LLM SETTINGS
            llm=openai.LLM(
                model="gpt-4.1-nano",    # Use the faster, newer model
                temperature=0.7,         # Slightly lower for faster generation
                max_tokens=1000,          # Shorter responses = faster processing
            ),
            
            # 🆕 STREAMING TTS WITH ELEVENLABS - Much faster response
            tts=elevenlabs.TTS(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Default voice or use your preferred voice
                model="eleven_turbo_v2_5",        # Fastest model
                streaming_latency=4,              # Max streaming optimization
                enable_ssml_parsing=False,        # Faster without SSML
            ),
            
            chat_ctx=initial_ctx,
            
            # 🆕 ONLY USE ACTUAL LIVEKIT FEATURES
            preemptive_synthesis=True,         # This exists and helps
            
            fnc_ctx=fnc_ctx,
        )
        print("✅ Assistant object created with ultra-fast settings.")
        
        # 🆕 IMPROVED EVENT LISTENERS with debouncing and speech buffer
        last_user_speech_time = 0
        last_agent_speech_time = 0
        speech_buffer_time = 0.2  # Reduced from 0.3 for faster response
        
        @assistant.on("user_speech_committed")
        def on_user_speech(msg):
            nonlocal last_user_speech_time
            current_time = asyncio.get_event_loop().time()
            # Debounce rapid fire events
            if current_time - last_user_speech_time > 0.5:
                print(f"🎤 USER SAID: {msg.content}")
                last_user_speech_time = current_time
            
        @assistant.on("agent_speech_committed") 
        def on_agent_speech(msg):
            nonlocal last_agent_speech_time
            current_time = asyncio.get_event_loop().time()
            if current_time - last_agent_speech_time > 0.5:
                print(f"🤖 BOT SAID: {msg.content}")
                last_agent_speech_time = current_time
            
        @assistant.on("user_started_speaking")
        def on_user_start():
            print("🎤 User started speaking...")
            
        @assistant.on("user_stopped_speaking")
        def on_user_stop():
            print("🎤 User stopped speaking.")
            
        @assistant.on("agent_started_speaking")
        def on_agent_start():
            print("🤖 Bot started speaking...")
            
        @assistant.on("agent_stopped_speaking")
        def on_agent_stop(): 
            print("🤖 Bot stopped speaking.")
            
    except Exception as e:
        print("❌ Error setting up assistant:", e)
        import traceback
        traceback.print_exc()
        return
    
    try:
        assistant.start(ctx.room)
        print("✅ Assistant started with ultra-fast settings.")
    except Exception as e:
        print("❌ Error starting assistant:", e)
        import traceback
        traceback.print_exc()
        return
    
    try:
        # 🆕 MINIMAL WELCOME MESSAGE for fastest start
        await asyncio.sleep(0.3)  # Even shorter wait
        await assistant.say("Hey! Can you hear me clearly?", allow_interruptions=True)
        print("🗣️ Assistant spoke the welcome message.")
    except Exception as e:
        print("❌ Error during welcome message:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))