"""llm.get_llm: builds an OpenRouter-backed ChatOpenAI, or fails loudly without a key."""
import pytest
from langchain_openai import ChatOpenAI

from llm import DEFAULT_MODEL, OPENROUTER_BASE_URL, get_llm


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
        get_llm()


def test_returns_chat_openai_pointed_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = get_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == DEFAULT_MODEL
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.openai_api_key.get_secret_value() == "test-key"
    assert llm.temperature == 0.3


def test_temperature_is_configurable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert get_llm(temperature=0).temperature == 0
