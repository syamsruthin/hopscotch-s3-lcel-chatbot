"""
Shared fixtures. The key piece is FakeChatModel: a real LangChain chat model that never
touches the network, so every chain in chains.py can be run end to end, offline and
deterministically.

How the fake answers:
- a TRIAGE call (system prompt starts with "You triage") returns triage_reply(customer_text)
- any other call returns "<ROUTE TAG> reply", where the tag is read from the
  "[ROUTE: ...]" header of the branch prompt (or "DIRECT" when there is none)
Every call's messages are recorded in .calls so tests can inspect what the model saw.
"""
import re
from typing import Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

import chains
from chains import SAMPLE_TICKETS

_ROUTE_TAG = re.compile(r"\[ROUTE: ([A-Z -]+)\]")


class FakeChatModel(BaseChatModel):
    triage_reply: Callable[[str], str] = lambda text: "other,Low"
    fail: bool = False
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _respond(self, messages: list[BaseMessage]) -> str:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("model is down")
        system = messages[0].content if messages[0].type == "system" else ""
        if system.startswith("You triage"):
            return self.triage_reply(messages[-1].content)
        tag = _ROUTE_TAG.search(system)
        return f"{tag.group(1) if tag else 'DIRECT'} reply"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._respond(messages)))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for i, word in enumerate(self._respond(messages).split(" ")):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word if i == 0 else " " + word))


# What a well-behaved model would answer for each sample ticket. Note that the safety and
# high-value tickets are triaged as plain defects: decide_route must still send them to a human.
SAMPLE_TRIAGE = dict(zip(SAMPLE_TICKETS.values(), [
    "genuine_defect,High",    # Defect: Dino-Print sneakers
    "policy_violation,Low",   # Policy: worn jacket
    "genuine_defect,High",    # Safety: button, toddler
    "genuine_defect,High",    # Value: INR 7,200
    "unclear,Medium",         # Unclear: faded dress
    "other,Low",              # General: Pune shipping
]))


def sample_triage(text: str) -> str:
    return SAMPLE_TRIAGE.get(text, "other,Low")


@pytest.fixture
def fake_llm():
    return FakeChatModel(triage_reply=sample_triage)


@pytest.fixture(autouse=True)
def clean_store():
    """Chat histories are module-global; give every test an empty store."""
    chains.store.clear()
    yield
    chains.store.clear()
