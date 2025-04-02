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
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDM1ODIwODUsImlzcyI6IkFQSUFCN1dScGpQUnVQMyIsIm5hbWUiOiJ0ZXN0LXVzZXIiLCJuYmYiOjE3NDM1ODE3ODUsInN1YiI6InRlc3QtdXNlciIsInZpZGVvIjp7InJvb20iOiJDbG9zZXJTaW11bGF0b3IiLCJyb29tSm9pbiI6dHJ1ZX19.X8kggwdMBGhZZVv3VYLIBkBsmHpG_JUZD_MAIiUVXV8"
    room_name = "CloserSimulator"

    print(f"📡 Connecting to LiveKit room '{room_name}' with token...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


    print("✅ Connected to room.")

    print("🔧 Setting up assistant...")
    fnc_ctx = AssistantFnc()

    try:
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=openai.STT(),
            llm=openai.LLM(),
            tts=openai.TTS(instructions=prospect_prompt),
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