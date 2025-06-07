import os
import asyncio
import json
import PyPDF2
from io import BytesIO
from openai import AsyncOpenAI
import aiohttp

# LiveKit imports
import livekit
from livekit import agents, rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai, silero, cartesia

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

# DEBUG: Log environment variables
print("🔍 DEBUG - Environment variables in Fly machine:")
print(f"SUPABASE_URL: {'✅ Set' if SUPABASE_URL else '❌ Missing'} - {SUPABASE_URL}")
print(f"SUPABASE_SERVICE_ROLE: {'✅ Set' if SUPABASE_KEY else '❌ Missing'} - {SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")
print(f"SESSION_ID: {'✅ Set' if os.getenv('SESSION_ID') else '❌ Missing'}")
print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
print(f"CARTESIA_API_KEY: {'✅ Set' if CARTESIA_API_KEY else '❌ Missing'}")

# Fetch token from Supabase
async def fetch_supabase_token() -> str:
    """Fetch token from Supabase using session ID"""
    try:
        session_id = os.getenv("SESSION_ID")
        print(f"🔍 DEBUG - Retrieved session ID: {session_id}")
        
        print("🔍 DEBUG - Fetching token from Supabase...")
        print("🔍 DEBUG - Starting Supabase token fetch...")
        
        url = f"{SUPABASE_URL}/rest/v1/livekit_tokens?token=eq.{session_id}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        print(f"🔍 DEBUG - Making request to: {url}")
        print(f"🔍 DEBUG - Headers: apikey={SUPABASE_KEY[:20]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                print(f"🔍 DEBUG - Response status: {response.status}")
                response_text = await response.text()
                print(f"🔍 DEBUG - Response text: {response_text[:200]}...")
                
                if response.status == 200:
                    data = json.loads(response_text)
                    if data and len(data) > 0:
                        token_data = data[0]
                        print(f"✅ SUCCESS - Retrieved token for room: {token_data.get('room')}, identity: {token_data.get('identity')}")
                        print("✅ DEBUG - Token fetch successful for room: sales-room")
                        return token_data.get('token')
                    else:
                        print("❌ ERROR - No token data found in response")
                        return None
                else:
                    print(f"❌ ERROR - HTTP {response.status}: {response_text}")
                    return None
                    
    except Exception as e:
        print(f"❌ ERROR - Token fetch failed: {str(e)}")
        return None

# Extract PDF content
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        print(f"📄 DEBUG - Starting PDF extraction: {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        print(f"✅ DEBUG - PDF extraction completed. Text length: {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"❌ ERROR - PDF extraction failed: {str(e)}")
        return ""

# Generate prospect prompt using GPT
async def generate_prospect_prompt(sales_content: str) -> str:
    """Generate a dynamic prospect persona using GPT"""
    try:
        print("💬 DEBUG - Starting GPT prospect prompt generation...")
        
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Prospect generation prompt
        system_prompt = f"""
        Based on the following sales training content, create a realistic B2B prospect persona for a sales simulation.

        Sales Content:
        {sales_content[:3000]}

        Create a detailed prospect profile in this exact format:

        **📌 Prospect Identity**
        **Name**: [First Last Name]
        **Age & Location**: [Age, City, State]
        **Business Name & Type**: [Company Name, Industry]
        **Monthly Revenue / Team Size / Industry Stage**: [Revenue/month, # employees, stage]
        **How they found the company**: [Discovery method]
        **What they've consumed**: [Content/touchpoints]
        **Lead Warmth**: [Cold/Warm/Hot]

        **🧠 Mindset & Goals**
        - Goal: [Primary business objective]
        - Reason for considering help now: [Urgency/catalyst]
        - Success: [What success looks like to them]

        **❗ Objections & Hesitations**
        - Objection: [Primary concern about working with you]
        - Fears/Pain: [Underlying fears or pain points]
        - Past experiences: [Relevant negative experiences]

        **🗣️ Conversation Behavior**
        - Behavior: [How they'll act on the call]
        - Openness to being sold: [Receptiveness level]
        - Tone, pace, style: [Communication style]

        **💬 Example Trigger Replies (Optional)**
        - "What are your goals?"
        *"[Example response]"*
        - "What stood out to you?"
        *"[Example response]"*
        - "What's holding you back?"
        *"[Example response]"*

        **🎯 Instruction for GPT Conversation Mode:**
        Act as this prospect through a full discovery call. Maintain the tone, skepticism level, and behavior described. Never break character and respond as [Name] would, sticking to his motivations, objections, and style. Keep the conversation direct and trust-focused.

        Make this prospect realistic, challenging (toughness level 5/10), and based on someone who would actually benefit from the sales training described.
        """
        
        print("🤖 Sending request to OpenAI for prospect prompt...")
        
        # Generate prospect using GPT
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt}
            ],
            max_tokens=2000,
            temperature=0.8
        )
        
        prospect_prompt = response.choices[0].message.content
        print("✅ Got prospect prompt from OpenAI")
        print(f"📝 Prompt length: {len(prospect_prompt)} characters")
        print("✅ DEBUG - GPT prospect prompt generated. Length:", len(prospect_prompt), "characters")
        return prospect_prompt
        
    except Exception as e:
        print(f"❌ ERROR - GPT prospect generation failed: {str(e)}")
        # Fallback prospect if GPT fails
        return """
        **📌 Prospect Identity**
        **Name**: Alex Johnson
        **Age & Location**: 35, Austin, TX
        **Business Name & Type**: TechFlow Solutions, SaaS Startup
        **Monthly Revenue / Team Size / Industry Stage**: $50k/month, 8 employees, early stage
        **How they found the company**: Google search
        **What they've consumed**: Free consultation call
        **Lead Warmth**: Warm

        **🧠 Mindset & Goals**
        Looking to scale the sales team and improve conversion rates.

        **❗ Objections & Hesitations**
        Concerned about cost and time investment.

        **🗣️ Conversation Behavior**
        Direct and to the point, skeptical but open to proven solutions.

        **🎯 Instruction for GPT Conversation Mode:**
        Act as Alex Johnson, a startup founder who is interested but cautious about sales training investments.
        """

# Modern Agent class with function tools
class ProspectAgent(agents.Agent):
    def __init__(self, prospect_prompt: str):
        super().__init__(
            instructions=prospect_prompt + "\n\nIMPORTANT: Never end the call unless explicitly asked. Stay in character and continue the conversation.",
        )
        print("✅ DEBUG - ProspectAgent created without auto-end function")

# Main entrypoint function
@agents.entrypoint
async def entrypoint(ctx: JobContext):
    try:
        print("🚀 DEBUG - Starting entrypoint function...")
        
        # Fetch token from Supabase
        token = await fetch_supabase_token()
        if not token:
            print("❌ CRITICAL - Failed to get token from Supabase!")
            return
        
        # Extract sales content from PDF
        sales_content = extract_pdf_text("assets/sales.pdf")
        if not sales_content:
            print("❌ WARNING - No PDF content extracted, using fallback")
            sales_content = "Sales training content about prospecting and closing deals."
        
        # Generate dynamic prospect prompt
        prospect_prompt = await generate_prospect_prompt(sales_content)
        
        print("🧠 GPT Persona Prompt:")
        print(prospect_prompt)
        print("=" * 60)
        
        # Connect to LiveKit room
        print("📡 DEBUG - Attempting to connect to LiveKit room: 'sales-room'")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        print("✅ DEBUG - Successfully connected to LiveKit room")
        
        print("🔧 DEBUG - Creating Agent and AgentSession with gpt-4.1-nano + Cartesia Sonic 2...")
        
        # Create ProspectAgent
        print("🔧 DEBUG - Creating ProspectAgent with instructions...")
        agent = ProspectAgent(prospect_prompt)
        print("✅ DEBUG - ProspectAgent created successfully")
        
        # Create VAD with more sensitive settings and logging
        print("🔧 DEBUG - Creating VAD with settings...")
        vad_instance = silero.VAD.load(
            min_speech_duration=0.1,    # 🔥 MORE SENSITIVE - detect shorter speech
            min_silence_duration=0.3,   # 🔥 SHORTER SILENCE - faster response  
            activation_threshold=0.4,   # 🔥 LOWER THRESHOLD - easier to trigger
            deactivation_threshold=0.2, # 🔥 LOWER THRESHOLD - easier to trigger
        )
        print("✅ DEBUG - VAD created successfully")
        
        # Create STT with logging
        print("🔧 DEBUG - Creating STT with Whisper...")
        stt_instance = openai.STT(
            model="whisper-1",
            language="en",
        )
        print("✅ DEBUG - STT created successfully")
        
        try:
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
            
        except Exception as e:
            print(f"❌ ERROR - Agent/Session setup failed: {e}")
            raise e
        
        # Create agent session with detailed logging
        print("🔧 DEBUG - Creating AgentSession with all components...")
        session = VoiceAssistant(
            vad=vad_instance,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
            chat_ctx=ChatContext(
                messages=[
                    ChatMessage(
                        role="system",
                        content=prospect_prompt + "\n\nIMPORTANT: Never end the call unless explicitly asked. Stay in character and continue the conversation.",
                    )
                ]
            ),
        )
        
        print("🔧 DEBUG - Adding event handlers...")
        
        # Try multiple event handler approaches
        try:
            @session.on("user_speech_committed")
            def on_user_speech_committed(text: str):
                print(f"🎤 DEBUG - User speech committed: '{text}'")
                print(f"🎤 DEBUG - Text length: {len(text)} characters")
            print("✅ DEBUG - user_speech_committed handler added")
        except Exception as e:
            print(f"❌ DEBUG - Failed to add user_speech_committed handler: {e}")
        
        try:
            @session.on("user_started_speaking")
            def on_user_started_speaking():
                print("🎤 DEBUG - User started speaking (VAD detected)")
            print("✅ DEBUG - user_started_speaking handler added")
        except Exception as e:
            print(f"❌ DEBUG - Failed to add user_started_speaking handler: {e}")
            
        try:
            @session.on("user_stopped_speaking")
            def on_user_stopped_speaking():
                print("🎤 DEBUG - User stopped speaking (VAD ended)")
            print("✅ DEBUG - user_stopped_speaking handler added")
        except Exception as e:
            print(f"❌ DEBUG - Failed to add user_stopped_speaking handler: {e}")
        
        try:
            @session.on("agent_started_speaking")
            def on_agent_started_speaking():
                print("🗣️ DEBUG - Agent started speaking")
            print("✅ DEBUG - agent_started_speaking handler added")
        except Exception as e:
            print(f"❌ DEBUG - Failed to add agent_started_speaking handler: {e}")
            
        try:
            @session.on("agent_stopped_speaking") 
            def on_agent_stopped_speaking():
                print("🗣️ DEBUG - Agent stopped speaking")
            print("✅ DEBUG - agent_stopped_speaking handler added")
        except Exception as e:
            print(f"❌ DEBUG - Failed to add agent_stopped_speaking handler: {e}")
        
        # Try alternative event names
        try:
            @session.on("speech_recognized")
            def on_speech_recognized(text: str):
                print(f"🎤 DEBUG - Speech recognized: '{text}'")
            print("✅ DEBUG - speech_recognized handler added")
        except Exception as e:
            print(f"❌ DEBUG - Failed to add speech_recognized handler: {e}")
            
        try:
            @session.on("user_transcript")
            def on_user_transcript(text: str):
                print(f"🎤 DEBUG - User transcript: '{text}'")
            print("✅ DEBUG - user_transcript handler added")
        except Exception as e:
            print(f"❌ DEBUG - Failed to add user_transcript handler: {e}")
        
        print("✅ DEBUG - AgentSession created successfully")
        print("🔧 DEBUG - Starting AgentSession...")
        session.start(ctx.room)
        print("✅ DEBUG - AgentSession started successfully")
        
        # Add a simple periodic log to show the session is running
        print("🔄 DEBUG - Session is now running and waiting for audio input...")
        print("🎤 DEBUG - Make sure your microphone is working and speak clearly...")
        
        # Generate welcome message with debugging
        print("🗣️ DEBUG - Preparing to speak welcome message...")
        try:
            welcome_response = await session.generate_reply(
                ChatMessage(
                    role="user",
                    content="Hello"
                )
            )
            print("🗣️ DEBUG - Calling session.generate_reply() for welcome message...")
            print("✅ DEBUG - Welcome message generate_reply() call completed")
        except Exception as e:
            print(f"❌ DEBUG - Error generating welcome message: {e}")
            # Fallback - just wait for user input
            print("🔄 DEBUG - Continuing without welcome message...")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR in entrypoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))