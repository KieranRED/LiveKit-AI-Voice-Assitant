# main.py
import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc

from utils import extract_pdf_text
from gpt_utils import get_prospect_prompt

load_dotenv()

async def entrypoint(ctx: JobContext):
    # 🔹 1. Load and parse your business sales PDF
    pdf_path = "assets/sales.pdf"  # Adjust this path
    business_pdf_text = extract_pdf_text(pdf_path)

    # 🔹 2. Define persona simulation inputs
    fit_strictness = "strict"
    objection_focus = "trust"
    toughness_level = 5
    call_type = "discovery"
    tone = "direct"

    # 🔹 3. Generate a custom prospect profile from GPT
    prospect_prompt = await get_prospect_prompt(
        fit_strictness,
        objection_focus,
        toughness_level,
        call_type,
        tone,
        business_pdf_text,
    )

    print("\n🧠 Prospect Simulation Prompt:\n")
    print(prospect_prompt)
    print("\n" + "="*80 + "\n")


    # 🔹 4. Create system prompt for LLM
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=prospect_prompt,
    )

    # 🔹 5. Connect to LiveKit and launch assistant
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today!", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
