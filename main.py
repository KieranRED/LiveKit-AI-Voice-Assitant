# main.py
import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc

from pdf_utils import extract_pdf_text
from gpt_utils import get_prospect_prompt

load_dotenv()

async def entrypoint(ctx: JobContext):
    print("🚀 Starting entrypoint...")

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

    print("📡 Connecting to LiveKit room...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print("✅ Connected to room.")

    print("🔧 Setting up assistant...")
    fnc_ctx = AssistantFnc()

    try:
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=openai.STT(),
            llm=openai.LLM(),
            tts=openai.TTS(),
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
        await assistant.say("Hey, how can I help you today!", allow_interruptions=True)
        print("🗣️ Assistant spoke the welcome message.")
    except Exception as e:
        print("❌ Error during welcome message:", e)



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
