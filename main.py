# main.py - UPDATED WITH OPENAI TTS
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
from livekit.plugins import openai, silero
from pdf_utils import extract_pdf_text
from gpt_utils import get_prospect_prompt
import openai as openai_client  # Import OpenAI client directly

async def generate_voice_instructions(prospect_prompt: str) -> str:
    """Generate TTS voice instructions based on prospect personality"""
    try:
        client = openai_client.AsyncOpenAI()  # Use direct OpenAI client
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": f"""Based on this prospect persona, generate concise voice instructions for a text-to-speech system. The instructions should describe how this person would speak - their tone, pace, energy level, accent, and speaking style.

PROSPECT PERSONA:
{prospect_prompt}

Generate 2-3 sentences describing how this person would sound when speaking. Focus on:
- Speaking pace (fast/slow/moderate)
- Energy level (high/low/calm/energetic) 
- Tone (confident/hesitant/friendly/professional/casual)
- Accent or regional speaking style if mentioned
- Personality traits that affect speech

Example: "Speak with a confident, fast-paced tone like a busy entrepreneur. Use a slightly elevated energy level with clear articulation. Sound professional but approachable."

Voice instructions:"""
            }],
            temperature=0.7,
            max_tokens=150
        )
        
        voice_instructions = response.choices[0].message.content.strip()
        print(f"🎭 Voice instructions: {voice_instructions}")
        return voice_instructions
        
    except Exception as e:
        print(f"⚠️ Failed to generate voice instructions: {e}")
        return "Speak in a natural, conversational tone with moderate pace and energy."

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")

# Environment check - removed Cartesia and ElevenLabs
print("🔍 Environment Check:")
print(f"SUPABASE_URL: {'✅' if SUPABASE_URL else '❌'}")
print(f"SUPABASE_SERVICE_ROLE: {'✅' if SUPABASE_KEY else '❌'}")
print(f"SESSION_ID: {'✅' if os.getenv('SESSION_ID') else '❌'}")
print(f"OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")

class ProspectAgent(Agent):
    def __init__(self, prospect_prompt: str):
        super().__init__(
            instructions=prospect_prompt + "\n\nYou must act as the prospect and the user will try to close you. Stay in character no matter what never break character and continue the conversation.",
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
        
        # Generate TTS voice instructions based on prospect personality
        print("🎭 Generating voice personality...")
        voice_instructions = await generate_voice_instructions(prospect_prompt)
        
        # Extract and display persona info
        lines = prospect_prompt.split('\n')
        name_line = next((line for line in lines if '**Name**' in line or '**Name:**' in line), "Unknown")
        business_line = next((line for line in lines if '**Business' in line), "Unknown Business")
        
        print("=" * 60)
        print(f"👤 {name_line}")
        print(f"👤 {business_line}")
        print(f"🎭 Voice Style: {voice_instructions}")
        print("=" * 60)
        
        # Connect to room
        print(f"📡 Connecting to room: {room_name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        print("✅ Connected to LiveKit")
        
        # Initialize components with OpenAI TTS
        print("🔧 Initializing AI components...")
        agent = ProspectAgent(prospect_prompt)
        
        vad_instance = silero.VAD.load(
            min_speech_duration=0.05,  # Reduced from 0.1 - detect speech faster
            min_silence_duration=0.15,  # Reduced from 0.3 - shorter silence before stopping
            prefix_padding_duration=0.05,  # Reduced from 0.1 - less padding
            activation_threshold=0.3,  # Reduced from 0.4 - more sensitive
        )
        
        stt_instance = openai.STT(model="whisper-1", language="en")
        
        # Optimize LLM for faster responses - removed max_tokens parameter
        llm_instance = openai.LLM(
            model="gpt-4.1-nano",  # Keeping the faster model as requested
            temperature=0.7,
        )
        
        # CHANGED: Using OpenAI GPT-4o Mini TTS with voice instructions
        tts_instance = openai.TTS(
            model="gpt-4o-mini-tts",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            instructions=voice_instructions,
        )
        
        # Create session
        session = AgentSession(
            vad=vad_instance,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
        )
        
        # Add voice event handlers with timing diagnostics
        conversation_count = [0]
        last_speech_end_time = [None]
        
        @session.on("speech_created")
        def on_speech_created(event):
            if hasattr(event, 'source'):
                if event.source == 'generate_reply':
                    # This is bot speech
                    conversation_count[0] += 1
                    if last_speech_end_time[0]:
                        delay = asyncio.get_event_loop().time() - last_speech_end_time[0]
                        print(f"🤖 BOT SPEAKING [{conversation_count[0]:02d}] (delay: {delay:.2f}s)")
                    else:
                        print(f"🤖 BOT SPEAKING [{conversation_count[0]:02d}]")
        
        @session.on("user_state_changed")
        def on_user_state_changed(event):
            if hasattr(event, 'new_state'):
                if event.new_state == 'speaking':
                    print("🎤 User started speaking...")
                elif event.new_state == 'listening' and hasattr(event, 'old_state') and event.old_state == 'speaking':
                    last_speech_end_time[0] = asyncio.get_event_loop().time()
                    print("🎤 User stopped speaking.")
        
        @session.on("user_input_transcribed") 
        def on_user_input_transcribed(event):
            if hasattr(event, 'transcript') and hasattr(event, 'is_final') and event.is_final:
                if last_speech_end_time[0]:
                    stt_delay = asyncio.get_event_loop().time() - last_speech_end_time[0]
                    print(f"🎤 USER SAID: {event.transcript} (STT delay: {stt_delay:.2f}s)")
                else:
                    print(f"🎤 USER SAID: {event.transcript}")
        
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