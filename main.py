# main.py - CLEANED UP VERSION WITH STRUCTURED LOGGING
import asyncio
import os
import requests
import logging
from datetime import datetime
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

# Configure structured logging - SUPPRESS DEBUG SPAM
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

# Silence noisy third-party loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING) 
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

def log_environment_check():
    """Log environment variable status in a clean format"""
    env_status = {
        "SUPABASE_URL": "✅" if SUPABASE_URL else "❌",
        "SUPABASE_SERVICE_ROLE": "✅" if SUPABASE_KEY else "❌", 
        "SESSION_ID": "✅" if os.getenv('SESSION_ID') else "❌",
        "OPENAI_API_KEY": "✅" if os.getenv('OPENAI_API_KEY') else "❌",
        "ELEVEN_API_KEY": "✅" if os.getenv('ELEVEN_API_KEY') else "❌",
        "CARTESIA_API_KEY": "✅" if CARTESIA_API_KEY else "❌"
    }
    
    logger.info("Environment Check: " + " | ".join([f"{k}: {v}" for k, v in env_status.items()]))

class ProspectAgent(Agent):
    def __init__(self, prospect_prompt: str):
        super().__init__(
            instructions=prospect_prompt + "\n\nIMPORTANT: Never end the call unless explicitly asked. Stay in character and continue the conversation.",
        )
        logger.info("ProspectAgent initialized successfully")

def fetch_token_from_supabase(session_id):
    """Fetch LiveKit token from Supabase with clean error handling"""
    logger.info(f"Fetching token for session: {session_id[:20]}...")
    
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
            raise ValueError("Token not found for session_id")
        
        token_data = data[0]
        logger.info(f"✅ Token retrieved | Room: {token_data['room']} | Identity: {token_data['identity']}")
        return token_data['token'], token_data['room'], token_data['identity']
        
    except Exception as e:
        logger.error(f"❌ Supabase token fetch failed: {e}")
        raise

class ConversationLogger:
    """Clean conversation logging without spam"""
    
    def __init__(self):
        self.conversation_count = 0
        
    def log_user_speech(self, text: str):
        """Log user speech cleanly"""
        self.conversation_count += 1
        logger.info(f"🎤 USER [{self.conversation_count:02d}]: {text}")
        
    def log_agent_speech(self, text: str):
        """Log agent speech cleanly"""
        logger.info(f"🤖 AGENT [{self.conversation_count:02d}]: {text}")
        
    def log_agent_thinking(self):
        """Log when agent is processing"""
        logger.info(f"🧠 AGENT: Processing response...")

def setup_session_handlers(session: AgentSession, conv_logger: ConversationLogger):
    """Setup essential event handlers without spam"""
    
    # Track conversation flow
    @session.on("user_input_transcribed") 
    def on_user_speech(event):
        if event.is_final and event.transcript.strip():
            conv_logger.log_user_speech(event.transcript)
    
    @session.on("conversation_item_added")
    def on_conversation_item(event):
        if event.item.role == "assistant" and not event.item.interrupted:
            conv_logger.log_agent_speech(" ".join(event.item.content))
    
    @session.on("agent_state_changed")
    def on_agent_state_change(event):
        if event.new_state == "thinking":
            conv_logger.log_agent_thinking()
    
    # Only log critical user state changes (remove debug level)
    # Remove these debug logs entirely to reduce noise
    
    logger.info("Event handlers configured")

async def entrypoint(ctx):
    """Main entrypoint with clean logging"""
    start_time = datetime.now()
    logger.info("🚀 Starting AI Sales Bot...")
    
    try:
        # Environment check
        log_environment_check()
        
        # Get session details
        session_id = os.getenv("SESSION_ID")
        logger.info(f"Session ID: {session_id[:20]}...")
        
        # Fetch token
        token, room_name, identity = fetch_token_from_supabase(session_id)
        
        # Load PDF content
        pdf_path = "assets/sales.pdf"
        logger.info(f"📄 Loading PDF: {pdf_path}")
        business_pdf_text = extract_pdf_text(pdf_path)
        logger.info(f"✅ PDF loaded ({len(business_pdf_text)} chars)")
        
        # Generate prospect prompt with shorter responses
        logger.info("🧠 Generating prospect persona...")
        prospect_prompt = await get_prospect_prompt(
            "strict", "trust", 5, "discovery", "direct", business_pdf_text
        )
        
        # Add speed optimization to the prompt
        prospect_prompt += "\n\nIMPORTANT: Keep responses very short (1 sentence, max 15 words) for natural conversation flow. Be direct and conversational."
        
        logger.info(f"✅ Persona generated ({len(prospect_prompt)} chars)")
        
        # Show prospect details (condensed)
        logger.info("=" * 60)
        # Extract just the key details for cleaner logging
        lines = prospect_prompt.split('\n')
        key_lines = [line for line in lines if any(keyword in line.lower() for keyword in 
                    ['name:', 'business:', 'revenue:', 'objection:', 'tone:'])][:5]
        for line in key_lines:
            logger.info(f"👤 {line.strip()}")
        logger.info("=" * 60)
        
        # Connect to LiveKit
        logger.info(f"📡 Connecting to room: {room_name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("✅ Connected to LiveKit")
        
        # Initialize components
        logger.info("🔧 Initializing AI components...")
        
        agent = ProspectAgent(prospect_prompt)
        
        vad_instance = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3, 
            prefix_padding_duration=0.1,
            activation_threshold=0.4,
        )
        
        stt_instance = openai.STT(model="whisper-1", language="en")
        llm_instance = openai.LLM(
            model="gpt-4.1-nano", 
            temperature=0.7,
        )
        
        tts_instance = cartesia.TTS(
            model="sonic-2",  # Keep Sonic-2 as requested
            voice="6f84f4b8-58a2-430c-8c79-688dad597532",
            speed=1.2,  # Faster speech for quicker playback
            encoding="pcm_s16le", 
            sample_rate=22050,  # Slightly lower than 24kHz for speed
            # Streaming is enabled by default in LiveKit
        )
        
        session = AgentSession(
            vad=vad_instance,
            stt=stt_instance, 
            llm=llm_instance,
            tts=tts_instance,
        )
        
        # Setup clean event handlers
        conv_logger = ConversationLogger()
        setup_session_handlers(session, conv_logger)
        
        logger.info("✅ Components initialized")
        
        # Start session
        logger.info("🎯 Starting conversation session...")
        await session.start(agent=agent, room=ctx.room)
        
        # Send welcome message
        await asyncio.sleep(0.5)
        await session.generate_reply(instructions="Greet the user by saying 'Hey! Can you hear me clearly?'")
        
        # Log session ready
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Sales bot ready! (startup: {elapsed:.1f}s)")
        logger.info("🗣️ Conversation active - user can now speak...")
        
        # Simple heartbeat (much less frequent)
        while True:
            await asyncio.sleep(60)  # Every minute instead of 30 seconds
            logger.info("💓 Bot active")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))