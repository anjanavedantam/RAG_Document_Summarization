from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_SYSTEM = "You are a helpful, accurate assistant."


def generate(prompt: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.2) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

    client = Groq(api_key=api_key.strip())
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()