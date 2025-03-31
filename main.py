# main.py
import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc
from livekit.agents import llm
from livekit import api, rtc
import jwt


import os


from pdf_utils import extract_pdf_text
from gpt_utils import get_prospect_prompt

load_dotenv()

async def main():
    print("🚀 Starting manual assistant...")

    # Load system prompt from file
    try:
        # Step 1: Load your business PDF
        pdf_path = "assets/sales.pdf"
        business_pdf_text = extract_pdf_text(pdf_path)

        # Step 2: Define simulation inputs (hardcoded or passed in later)
        fit_strictness = "strict"
        objection_focus = "price"
        toughness_level = 5
        call_type = "discovery"
        tone = "laid-back"

        # Step 3: Generate the prompt from OpenAI
        prompt = await get_prospect_prompt(
            fit_strictness,
            objection_focus,
            toughness_level,
            call_type,
            tone,
            business_pdf_text,
        )
    except FileNotFoundError:
        print("❌ Error: latest_prompt.txt not found.")
        return

    # Set up chat context
    chat_ctx = llm.ChatContext().append(
        role="system",
        text=prompt,
    )

    # Load assistant components
    fnc_ctx = AssistantFnc()

    # Connect to LiveKit
    livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    room_name = os.getenv("LIVEKIT_ROOM", "playground-mCa2-j17e")
    identity = "kieran"  # or make dynamic

    if not api_key or not api_secret:
        print("❌ LIVEKIT_API_KEY or SECRET not set")
        return

    grants = api.VideoGrants(room_join=True, room=room_name)
    



    access_token = api.AccessToken(
        api_key=api_key,
        api_secret=api_secret,
    )

    access_token.identity = identity  # ✅ Set identity separately
    access_token.video = api.VideoGrants(room_join=True, room=room_name)


    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDM1MTIxNzIsImlzcyI6IkFQSUFCN1dScGpQUnVQMyIsIm5hbWUiOiJ0ZXN0LXVzZXIiLCJuYmYiOjE3NDM1MTE4NzIsInN1YiI6InRlc3QtdXNlciIsInZpZGVvIjp7InJvb20iOiJwbGF5Z3JvdW5kLW1DYTItajE3ZSIsInJvb21Kb2luIjp0cnVlfX0.ACAdm74Klmw8k-cl2jCFkX1jyq881OVaFqgyeFFL8Lc"

    print("🔐 JWT Token:\n", token)
    print("🏷️ Room Name in Grant:", room_name)

    decoded = jwt.decode(token, options={"verify_signature": False})
    print("🧠 Decoded Token Payload:")
    print(decoded)


    print(f"📡 Connecting to {livekit_url}...")
    room = rtc.Room()
    await room.connect(
        url=livekit_url,
        token=token,
    )



    print("✅ Connected to room:", room.name)

    # Start assistant
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(instructions=prompt),
        chat_ctx=chat_ctx,
        fnc_ctx=fnc_ctx,
    )
    assistant.start(room)
    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today!", allow_interruptions=True)
    print("🗣️ Assistant spoke welcome message.")

    # Keep running
    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
