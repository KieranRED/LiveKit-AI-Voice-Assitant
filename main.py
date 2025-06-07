import asyncio
import os
import logging
import requests
import traceback
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import openai, silero, cartesia, elevenlabs
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProspectAgent(Agent):
    def __init__(self, prospect_prompt: str):
        super().__init__(
            instructions=prospect_prompt + "\n\nIMPORTANT: Keep responses very short (1 sentence, max 10-15 words) for natural conversation flow. Be direct and conversational.",
        )
        logger.info("ProspectAgent initialized successfully")

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise

async def get_prospect_prompt(fit_strictness, objection_focus, toughness_level, call_type, tone, business_content):
    """Generate dynamic prospect persona using OpenAI"""
    import openai as openai_client
    
    logger.info("🤖 Sending request to OpenAI for prospect prompt...")
    
    try:
        client = openai_client.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": f"""Create a realistic sales prospect persona for role-playing training. 

                Business Context: {business_content[:3000]}
                
                Generate a prospect with:
                - Fit Strictness: {fit_strictness}
                - Main Objection Focus: {objection_focus} 
                - Toughness Level: {toughness_level}/10
                - Call Type: {call_type}
                - Communication Tone: {tone}
                
                Create a short, realistic persona with:
                1. Name and role
                2. Primary objection (based on objection focus)
                3. DISC profile hints for response style
                4. Specific pain points related to the business
                5. Buying behavior patterns
                
                Format as role-playing instructions for an AI to act as this prospect.
                Keep it under 2000 characters - concise but detailed enough for realistic interaction."""
            }],
            temperature=0.8,
            max_tokens=500
        )
        
        prompt = response.choices[0].message.content
        logger.info("✅ Got prospect prompt from OpenAI")
        logger.info(f"📝 Prompt length: {len(prompt)} characters")
        return prompt
        
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        raise

def fetch_token_from_supabase(session_id):
    """Fetch LiveKit token from Supabase"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE")
    
    url = f"{supabase_url}/rest/v1/livekit_tokens?token=eq.{session_id}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError("Token not found for session_id")
        
        token_data = data[0]
        return token_data['token'], token_data['room'], token_data['identity']
        
    except Exception as e:
        logger.error(f"Supabase token fetch failed: {e}")
        raise

async def entrypoint(job_ctx: JobContext):
    """Main entrypoint function for the sales bot"""
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Environment check
        logger.info("🚀 Starting AI Sales Bot...")
        required_envs = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE", "SESSION_ID", "OPENAI_API_KEY", "ELEVEN_API_KEY", "CARTESIA_API_KEY"]
        env_status = " | ".join([f"{env}: {'✅' if os.getenv(env) else '❌'}" for env in required_envs])
        logger.info(f"Environment Check: {env_status}")
        
        # Get session and fetch token
        session_id = os.getenv("SESSION_ID")
        logger.info(f"Session ID: {session_id[:20]}...")
        logger.info(f"Fetching token for session: {session_id[:20]}...")
        
        token, room_name, identity = fetch_token_from_supabase(session_id)
        logger.info(f"✅ Token retrieved | Room: {room_name} | Identity: {identity}")
        
        # Load and process PDF
        pdf_path = "assets/sales.pdf"
        logger.info(f"📄 Loading PDF: {pdf_path}")
        business_content = extract_pdf_text(pdf_path)
        logger.info(f"✅ PDF loaded ({len(business_content)} chars)")
        
        # Generate dynamic prospect persona
        logger.info("🧠 Generating prospect persona...")
        prospect_prompt = await get_prospect_prompt(
            fit_strictness="moderate",
            objection_focus="price",
            toughness_level=6,
            call_type="discovery",
            tone="professional",
            business_content=business_content
        )
        logger.info(f"✅ Persona generated ({len(prospect_prompt)} chars)")
        
        # Display persona info
        logger.info("=" * 60)
        # Extract key info for display (basic parsing)
        lines = prospect_prompt.split('\n')
        for line in lines[:3]:
            if 'name' in line.lower() or 'objection' in line.lower():
                logger.info(f"👤 {line.strip()}")
        logger.info("=" * 60)
        
        # Connect to LiveKit room
        logger.info(f"📡 Connecting to room: {room_name}")
        await job_ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("✅ Connected to LiveKit")
        
        # Initialize AI components
        logger.info("🔧 Initializing AI components...")
        
        # Create prospect agent
        agent = ProspectAgent(prospect_prompt)
        
        # Initialize VAD with sensitive settings
        vad_instance = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            prefix_padding_duration=0.1,
            activation_threshold=0.4,
        )
        
        # Initialize STT
        stt_instance = openai.STT(model="whisper-1", language="en")
        
        # Initialize LLM with debugging
        logger.info("🧠 Initializing OpenAI LLM...")
        llm_instance = openai.LLM(
            model="gpt-4.1-nano", 
            temperature=0.7,
        )
        logger.info("✅ LLM initialized successfully")
        
        # Initialize TTS with debugging
        logger.info("🔊 Initializing Cartesia TTS...")
        try:
            tts_instance = cartesia.TTS(
                model="sonic-2",
                voice="6f84f4b8-58a2-430c-8c79-688dad597532",
                speed=1.2,
                encoding="pcm_s16le", 
                sample_rate=22050,
            )
            logger.info("✅ Cartesia TTS initialized successfully")
        except Exception as e:
            logger.error(f"❌ Cartesia TTS failed: {e}")
            logger.info("🔄 Falling back to ElevenLabs TTS...")
            tts_instance = elevenlabs.TTS()
            logger.info("✅ ElevenLabs TTS initialized as fallback")
        
        # Create AgentSession with all components
        session = AgentSession(
            vad=vad_instance,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
        )
        
        # Add debug event handlers
        @session.on("user_speech_committed")
        def on_user_speech_committed(text: str):
            logger.info(f"🎤 USER [{len([])}{len([])+1:02d}]: {text}")
        
        @session.on("user_started_speaking")
        def on_user_started_speaking():
            logger.info("🎤 User started speaking")
            
        @session.on("user_stopped_speaking")
        def on_user_stopped_speaking():
            logger.info("🎤 User stopped speaking")
        
        @session.on("agent_started_speaking")
        def on_agent_started_speaking():
            logger.info("🔊 Agent started speaking (TTS active)")
            
        @session.on("agent_stopped_speaking")
        def on_agent_stopped_speaking():
            logger.info("🔇 Agent stopped speaking (TTS complete)")

        logger.info("Event handlers configured")
        logger.info("✅ Components initialized")
        
        # Start the session
        logger.info("🎯 Starting conversation session...")
        await session.start(agent=agent, room=job_ctx.room)
        
        # Generate welcome message
        logger.info("🧠 AGENT: Processing response...")
        await session.generate_reply(instructions="Greet the user by saying 'Hey! Can you hear me clearly?'")
        
        # Calculate startup time
        startup_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"🎉 Sales bot ready! (startup: {startup_time:.1f}s)")
        logger.info("🗣️ Conversation active - user can now speak...")
        
        # Keep session alive with heartbeat
        while True:
            await asyncio.sleep(30)
            logger.info("💓 Session heartbeat - active and listening...")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))