import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a finance research agent. 
Provide precise, sourced data when available.
"""

def ask_gpt(user_message: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-5-mini",   # <-- adjust if your access uses a different name, e.g. "gpt-4o"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=1,
    )
    return resp.choices[0].message.content

def extract_ticker(user_prompt: str) -> str:
    """Ask GPT-5 to identify the most relevant ticker symbol (S&P 500 only)."""
    system = (
        "You are a financial assistant. "
        "When given a user request, return ONLY the ticker symbol "
        "of the relevant S&P 500 company. "
        "If none applies, return 'NONE'."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.0,
    )
    ticker = resp.choices[0].message.content.strip().upper()
    return None if ticker == "NONE" else ticker



