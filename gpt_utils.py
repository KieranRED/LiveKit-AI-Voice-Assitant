# gpt_utils.py
import openai
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_prospect_prompt(fit_strictness, objection_focus, toughness_level, call_type, tone, business_pdf_text):
    prompt = f"""
You're a Prospect Simulator GPT.

Your role is to simulate realistic, dynamic prospects for sales call roleplays based on a business's sales PDF.

You generate a new, unique persona each time, tailored to the business's offer, client type, and sales process — complete with motivations, objections, and specific behavior in the call.

📥 INPUTS:
Business PDF → Sales process, target customer, and offer info from the company's perspective.

[Fit Strictness]: {fit_strictness}
[Objection Focus]: {objection_focus}
[Toughness Level]: {toughness_level}
[Call Type]: {call_type}
[Tone]: {tone}

✅ OUTPUT STRUCTURE
Return the result as a Markdown-formatted prompt that can be copied into ChatGPT's conversation mode.

📌 Prospect Identity
Name
Age & Location
Business Name & Type (if relevant)
Monthly Revenue / Team Size / Industry Stage
How they found the company
What they've consumed (free course, audit, etc.)
Lead Warmth (cold, warm, hot)

🧠 Mindset & Goals
What they're trying to fix or improve
Why they're considering help now
What success looks like for them

❗ Objections & Hesitations
Chosen or auto-generated objection (based on input)
Any personal skepticism, fears, or pain points
Past experiences that make them cautious

🗣️ Conversation Behavior
How they'll behave on the call (based on toughness level)
How open they are to being sold
Their tone, pace, and style (from tone input)

💬 Example Trigger Replies (Optional)
Include 2–3 short sample responses to common sales questions:
"What are your goals?"
"What stood out to you?"
"What's holding you back?"

🎯 Instruction for GPT Conversation Mode:
Act as this prospect through a full call. Stick to the tone, objection, and difficulty level. Never break character.
"""
    
    print("🤖 Sending request to OpenAI for prospect prompt...")
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # 🆕 Changed from gpt-4.1-nano to a valid model
            messages=[{"role": "system", "content": prompt}],
        )
        
        result = response.choices[0].message.content
        print("✅ Got prospect prompt from OpenAI")
        print(f"📝 Prompt length: {len(result)} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Error getting prospect prompt: {e}")
        # Fallback prompt if OpenAI fails
        fallback_prompt = """
You are Sarah, a 32-year-old marketing consultant from Austin, Texas. 
You run a small digital marketing agency with 3 employees making about $15k/month. 
You're interested in scaling but skeptical about high-ticket coaching programs.
You're direct, ask tough questions, and won't be easily sold to.
Keep responses conversational and realistic.
"""
        print("🔄 Using fallback prompt")
        return fallback_prompt