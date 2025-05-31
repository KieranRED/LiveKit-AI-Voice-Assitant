# main.py
import asyncio
import os
import requests
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc

from pdf_utils import extract_pdf_text
from gpt_utils import get_prospect_prompt

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")  # Use service role or env-safe key


def fetch_token_from_supabase(session_id):
    url = f"{SUPABASE_URL}/rest/v1/livekit_tokens?id=eq.{session_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }

    res = requests.get(url, headers=headers)
    res.raise_for_status()
    data = res.json()
    if not data:
        raise ValueError("❌ Token not found for session_id")

    return data[0]['token'], data[0]['room'], data[0]['identity']


async def entrypoint(ctx: JobContext):
    print("🚀 Starting entrypoint...")

    session_id = os.getenv("SESSION_ID")  # Must be passed into environment or injected
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
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY, token=token)

    print("✅ Connected to room.")

    print("🔧 Setting up assistant...")
    fnc_ctx = AssistantFnc()

    try:
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=openai.STT(),
            llm=openai.LLM(),
            tts=openai.TTS(instructions=prospect_prompt.strip()),
            chat_ctx=initial_ctx,
            fnc_ctx=fnc_ctx,
        )
        print("✅ Assistant object created.")
    except Exception as e:
        print("❌ Error setting up assistant:", e)
        return

    try:
        assistant.start(ctx.room)
        print("✅ Assistant started.")
    except Exception as e:
        print("❌ Error starting assistant:", e)
        return

    try:
        await asyncio.sleep(1)
        await assistant.say("Hey", allow_interruptions=True)
        print("🗣️ Assistant spoke the welcome message.")
    except Exception as e:
        print("❌ Error during welcome message:", e)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
