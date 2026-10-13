"""prompts.py: the policy is loaded and every template formats with the inputs the chains supply."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from prompts import (
    DEFECT_PROMPT,
    ESCALATION_PROMPT,
    GENERAL_PROMPT,
    POLICY,
    POLICY_REPLY_PROMPT,
    TRIAGE_PROMPT,
)

BRANCH_PROMPTS = {
    "HUMAN HAND-OFF": ESCALATION_PROMPT,
    "GENUINE DEFECT": DEFECT_PROMPT,
    "POLICY EXPLANATION": POLICY_REPLY_PROMPT,
    "GENERAL": GENERAL_PROMPT,
}


def test_policy_file_is_loaded():
    assert "HOPSCOTCH RETURN AND REFUND POLICY" in POLICY
    assert "MANDATORY HUMAN REVIEW" in POLICY


def test_triage_prompt_asks_for_category_priority_and_passes_input_through():
    system, human = TRIAGE_PROMPT.format_messages(input="Do you ship to Pune?")
    for category in ("genuine_defect", "policy_violation", "unclear", "other"):
        assert category in system.content
    assert "category,priority" in system.content
    assert human.content == "Do you ship to Pune?"


@pytest.mark.parametrize("tag, prompt", BRANCH_PROMPTS.items(), ids=list(BRANCH_PROMPTS))
def test_branch_prompt_structure(tag, prompt):
    history = [HumanMessage("hi"), AIMessage("hello!")]
    messages = prompt.format_messages(input="Where is my order?", chat_history=history,
                                      triage_summary="category=other; priority=Low")
    system, *middle, human = messages

    assert system.content.startswith(f"[ROUTE: {tag}]")
    assert POLICY in system.content, "policy text should be injected via .partial()"
    assert "NEVER mention" in system.content
    assert middle == history, "chat history should sit between the system and human turns"
    assert human.content == "[internal triage: category=other; priority=Low]\nCustomer: Where is my order?"


@pytest.mark.parametrize("prompt", BRANCH_PROMPTS.values(), ids=list(BRANCH_PROMPTS))
def test_branch_prompts_only_need_runtime_inputs(prompt):
    assert set(prompt.input_variables) == {"input", "chat_history", "triage_summary"}
