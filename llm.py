"""
llm.py - one place that knows how to build a chat model.

We talk to OpenRouter (an OpenAI-compatible gateway) through LangChain's ChatOpenAI class, so switching from GPT to Claude to Gemini to Llama is
just a different model-name string - nothing else in the app changes.
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from a local .env file (ignored by git)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Return a ChatOpenAI pointed at OpenRouter, always using DEFAULT_MODEL."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set (see .env.example).")
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )