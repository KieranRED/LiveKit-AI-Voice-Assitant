#!/usr/bin/env python

import asyncio
import logging
import os
from typing import Annotated

import aiohttp
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext, 
    WorkerOptions, 
    cli, 
    tokenize, 
    tts
)
# Note: ChatContext is likely in llm module now, but we'll use Agent instructions instead
from livekit.plugins import openai, silero
from pdf_utils import extract_pdf_text
from gpt_utils import get_prospect_prompt
import openai as openai_client  # Import OpenAI client directly

logger = logging.getLogger("voice-assistant")

async def generate_voice_instructions(prospect_prompt: str) -> str:
    """Generate TTS voice instructions based on prospect personality"""
    try:
        client = openai_client.AsyncOpenAI()  # Use direct OpenAI client
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": f"""Based on this prospect persona, generate concise voice instructions for a text-to-speech system. The instructions should describe how this person would sound when speaking - their tone, pace, energy level, accent, and speaking style.

PROSPECT PERSONA:
{prospect_prompt}

Generate 2-3 sentences describing how this person would sound when speaking. Focus on:
- Speaking pace (fast/slow/moderate)
- Energy level (high/low/calm/energetic) 
- Tone (confident/hesitant/friendly/professional/casual)
- Regional accent if mentioned in persona (Southern, New York, Midwest, etc.)
- Personality traits that affect speech
- Always be specific about accent and speaking style for consistency

Be very explicit about accent and vocal characteristics to ensure consistent TTS output.

IMPORTANT: If the persona mentions a location or region, ALWAYS include the appropriate accent (Texas=Southern, New York=NYC, California=West Coast, etc.). Be very specific like "strong Southern drawl" or "crisp New York accent" for better consistency.

Voice instructions:"""
            }],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate voice instructions: {e}")
        return "Speak in a natural, conversational tone with moderate pace and energy."

def prewarm(proc: agents.JobProcess):
    """Preload models and resources"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main agent entry point"""
    
    logger.info("🔍 Environment Check:")
    logger.info(f"SUPABASE_URL: {'✅' if os.getenv('SUPABASE_URL') else '❌'}")
    logger.info(f"SUPABASE_SERVICE_ROLE: {'✅' if os.getenv('SUPABASE_SERVICE_ROLE') else '❌'}")
    logger.info(f"SESSION_ID: {'✅' if os.getenv('SESSION_ID') else '❌'}")
    logger.info(f"OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")

    # Connect to the room first (required in v1.0+)
    print("📡 Connecting to LiveKit...")
    await ctx.connect()
    print("✅ Connected to LiveKit")

    # Now wait for a participant to connect
    await ctx.wait_for_participant()

    print("🚀 Starting AI Sales Bot...")

    # Get session token from environment variable
    session_token = os.getenv("SESSION_ID")
    if not session_token:
        raise ValueError("SESSION_ID environment variable is required")
    
    print(f"🔍 Fetching token for session: {session_token}")

    try:
        # Fetch token from Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_service_role = os.getenv("SUPABASE_SERVICE_ROLE")
        
        if not supabase_url or not supabase_service_role:
            raise ValueError("Supabase configuration missing")

        async with aiohttp.ClientSession() as session:
            headers = {
                "apikey": supabase_service_role,
                "Authorization": f"Bearer {supabase_service_role}",
                "Content-Type": "application/json"
            }
            
            async with session.get(
                f"{supabase_url}/rest/v1/sessions?session_token=eq.{session_token}&select=*",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        session_data = data[0]
                        room_name = session_data.get("room_name", "sales-room")
                        identity = session_data.get("identity", "user1234")
                        print(f"✅ Token retrieved | Room: {room_name} | Identity: {identity}")
                    else:
                        raise ValueError("Session not found")
                else:
                    raise ValueError(f"Failed to fetch session: {response.status}")

    except Exception as e:
        logger.error(f"❌ Error fetching session: {e}")
        # Fallback values
        room_name = "sales-room"
        identity = "user1234"

    try:
        # Load and process PDF
        print("📄 Loading PDF: assets/sales.pdf")
        pdf_content = extract_pdf_text("assets/sales.pdf")
        print(f"✅ PDF loaded ({len(pdf_content)} chars)")
        
        # Generate prospect persona with required arguments
        # Adjust these parameters based on your gpt_utils.py function requirements
        print("🧠 Generating prospect persona...")
        prospect_prompt = await get_prospect_prompt(
            pdf_content,
            objection_focus="price_concerns",  # Common: "price_concerns", "time_concerns", "authority_concerns"
            toughness_level="medium",         # Options: "easy", "medium", "hard"
            call_type="discovery",           # Options: "discovery", "demo", "closing", "follow_up"
            tone="professional",             # Options: "professional", "casual", "friendly", "formal"
            business_pdf_text=pdf_content
        )
        print(f"📝 Prompt length: {len(prospect_prompt)} characters")
        
        # Generate voice personality based on prospect
        print("🎭 Generating voice personality...")
        voice_instructions = await generate_voice_instructions(prospect_prompt)
        print(f"🎭 Voice instructions: {voice_instructions}")
        
        # Extract key details for display
        lines = prospect_prompt.split('\n')
        name_line = next((line for line in lines if 'Name:' in line), "Name: Unknown")
        business_line = next((line for line in lines if 'Business Name' in line or 'Company' in line), "Business: Unknown")
        
        print("=" * 60)
        print(f"👤 - {name_line}")
        print(f"👤 - {business_line}")
        print(f"🎭 Voice Style: {voice_instructions}")
        print("=" * 60)
        
        # Update system context with prospect info
        system_prompt = f"""You are an AI sales agent speaking to a prospect over voice chat. Here's your prospect info:

{prospect_prompt}

VOICE STYLE: {voice_instructions}

Your goal is to have a natural conversation and determine if they're a good fit for our services. Always:
- Keep responses conversational and brief (1-2 sentences max)
- Use their name when appropriate 
- Ask follow-up questions to understand their needs
- Be helpful but not pushy
- Match the voice style described above
- Sound natural and human-like"""

    except Exception as e:
        logger.error(f"❌ Error setting up prospect: {e}")
        # Fallback to basic context
        voice_instructions = "Speak in a natural, conversational tone with moderate pace and energy."
        system_prompt = "You are a voice assistant created by LiveKit. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation."

    # Already connected above

    # Initialize components with optimizations
    print("🔧 Initializing AI components...")
    try:
        # Optimized VAD settings for faster response
        vad_instance = silero.VAD.load(
            min_silence_duration=0.15,  # Faster silence detection (was 0.3)
            min_speech_duration=0.05,   # Faster speech detection (was 0.1)  
            activation_threshold=0.3,   # More sensitive (was 0.4)
        )
        
        # TTS with correct model
        tts_instance = openai.TTS(
            model="gpt-4o-mini-tts",  # Fixed TTS model
            voice="alloy",
            # voice_instructions=voice_instructions,  # This parameter may not exist in v1.0+
        )
        
        stt_instance = openai.STT(model="whisper-1", language="en")
        
        # Optimize LLM for faster responses with correct model
        llm_instance = openai.LLM(
            model="gpt4.1nano",  # Fixed LLM model
            temperature=0.7,
        )

        # Create Agent with system prompt
        agent = Agent(instructions=system_prompt)

        # Create AgentSession with optimized settings
        session = AgentSession(
            vad=vad_instance,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
        )

        # Conversation tracking variables
        conversation_count = [0]
        last_speech_end_time = [None]
        welcome_sent = [False]
        
        # Event handlers for tracking conversation flow (v1.0+ event names)
        
        # Listen for user speech transcription
        @session.on("user_input_transcribed")
        def on_user_input_transcribed(event):
            if hasattr(event, 'transcript') and hasattr(event, 'is_final') and event.is_final:
                if last_speech_end_time[0]:
                    stt_delay = asyncio.get_event_loop().time() - last_speech_end_time[0]
                    print(f"🎤 USER SAID: {event.transcript} (STT delay: {stt_delay:.2f}s)")
                else:
                    print(f"🎤 USER SAID: {event.transcript}")
        
        # Listen for agent speech creation
        @session.on("speech_created")
        def on_speech_created(event):
            if not welcome_sent[0]:
                welcome_sent[0] = True
                print("🤖 BOT SPEAKING [WELCOME] (greeting)")
                return
                
            conversation_count[0] += 1
            if last_speech_end_time[0]:
                delay = asyncio.get_event_loop().time() - last_speech_end_time[0]
                print(f"🤖 BOT SPEAKING [ACTUAL-{conversation_count[0]:02d}] (delay: {delay:.2f}s)")
            else:
                print(f"🤖 BOT SPEAKING [ACTUAL-{conversation_count[0]:02d}]")
        
        # Listen for user state changes (speaking/not speaking)
        @session.on("user_state_changed")
        def on_user_state_changed(event):
            if hasattr(event, 'state'):
                if event.state == 'speaking':
                    print("🎤 User started speaking...")
                elif event.state == 'listening':
                    last_speech_end_time[0] = asyncio.get_event_loop().time()
                    print("🎤 User stopped speaking.")
        
        # Listen for conversation items being added to chat history
        @session.on("conversation_item_added")
        def on_conversation_item_added(event):
            if hasattr(event, 'item'):
                if event.item.role == 'user':
                    print(f"📝 User message added to chat: {event.item.text_content}")
                elif event.item.role == 'assistant':
                    print(f"📝 Bot message added to chat: {event.item.text_content}")

        print("🔧 Speech event handlers added")

        # Start the session
        print("🔧 Starting session...")
        await session.start(agent=agent, room=ctx.room)

        # Send welcome message
        print("🗣️ Sending welcome message...")
        await asyncio.sleep(0.5)
        await session.generate_reply(instructions="Greet the user by saying 'Hey! Can you hear me clearly?'")
        
        print("🎉 Sales bot ready! Conversation active...")

        # Keep the session alive
        while True:
            await asyncio.sleep(30)
            print("💓 Bot running...")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )