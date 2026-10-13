"""The LCEL chains, run end to end against FakeChatModel (see conftest.py)."""
import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory

import chains
from chains import (
    SAMPLE_EXPECTED,
    SAMPLE_TICKETS,
    build_bot,
    build_enrichment_pipeline,
    build_full_chain,
    build_greeting_chains,
    build_route_branch,
    build_signals,
    build_triage_subchain,
    decide_route,
    explain_route,
    get_session_history,
)
from conftest import FakeChatModel

ROUTE_REPLIES = {
    "human_review": "HUMAN HAND-OFF reply",
    "defect": "GENUINE DEFECT reply",
    "policy": "POLICY EXPLANATION reply",
    "general": "GENERAL reply",
}
DEFECT_TEXT = SAMPLE_TICKETS["Defect: Dino-Print sneakers"]


def session(sid):
    return {"configurable": {"session_id": sid}}


# --------------------------------------------------------------------------- triage + signals
def test_triage_subchain_parses_messy_model_output():
    llm = FakeChatModel(triage_reply=lambda text: "  Category: Genuine Defect , Priority: HIGH ")
    assert build_triage_subchain(llm).invoke({"input": "sole peeled"}) == {
        "category": "genuine_defect", "priority": "High"}


def test_signals_combines_llm_and_deterministic_checks(fake_llm):
    assert build_signals(fake_llm).invoke(DEFECT_TEXT) == {
        "triage": {"category": "genuine_defect", "priority": "High"},
        "safety": {"flag": False, "reason": None},
        "order_value": None,
        "order_id": "HS88213",
    }
    assert len(fake_llm.calls) == 1, "only the triage step calls the model"


def test_sequential_enrichment_produces_the_same_signals(fake_llm):
    sequential = build_enrichment_pipeline(fake_llm).invoke({"input": DEFECT_TEXT})
    parallel = build_signals(fake_llm).invoke(DEFECT_TEXT)
    assert sequential == {"input": DEFECT_TEXT, **parallel}


def test_signals_batch_over_all_samples(fake_llm):
    results = build_signals(fake_llm).batch(list(SAMPLE_TICKETS.values()), config={"max_concurrency": 3})
    assert [r["triage"]["category"] for r in results] == [
        "genuine_defect", "policy_violation", "genuine_defect", "genuine_defect", "unclear", "other"]


# --------------------------------------------------------------------------- RunnableBranch
@pytest.mark.parametrize("needs_human, category, expected", [
    (True, "genuine_defect", "HUMAN HAND-OFF reply"),   # needs_human is checked first
    (False, "genuine_defect", "GENUINE DEFECT reply"),
    (False, "policy_violation", "POLICY EXPLANATION reply"),
    (False, "other", "GENERAL reply"),
    (False, "unclear", "GENERAL reply"),                # default branch
])
def test_route_branch_picks_the_reply_prompt(fake_llm, needs_human, category, expected):
    routed = {"input": "hi", "chat_history": [], "triage_summary": "s",
              "needs_human": needs_human, "category": category}
    assert build_route_branch(fake_llm).invoke(routed) == expected


def test_branch_prompt_carries_the_internal_triage_note(fake_llm):
    routed = decide_route({"input": DEFECT_TEXT, "chat_history": [],
                           "signals": build_signals(fake_llm).invoke(DEFECT_TEXT)})
    build_route_branch(fake_llm).invoke(routed)
    human_turn = fake_llm.calls[-1][-1].content
    assert human_turn.startswith("[internal triage: category=genuine_defect; priority=High")
    assert "order_id=HS88213" in human_turn
    assert human_turn.endswith(f"Customer: {DEFECT_TEXT}")


# --------------------------------------------------------------------------- full chain
@pytest.mark.parametrize("name", list(SAMPLE_TICKETS))
def test_full_chain_routes_every_sample_ticket_as_expected(fake_llm, name):
    reply = build_full_chain(fake_llm).invoke({"input": SAMPLE_TICKETS[name], "chat_history": []})
    assert reply == ROUTE_REPLIES[SAMPLE_EXPECTED[name]]


def test_full_chain_makes_two_model_calls(fake_llm):
    build_full_chain(fake_llm).invoke({"input": "Do you ship to Pune?", "chat_history": []})
    assert len(fake_llm.calls) == 2  # triage + reply


def test_garbage_triage_output_is_escalated_to_a_human():
    llm = FakeChatModel(triage_reply=lambda text: "I'd rather not say")
    reply = build_full_chain(llm).invoke({"input": "Do you ship to Pune?", "chat_history": []})
    assert reply == "HUMAN HAND-OFF reply"


def test_full_chain_propagates_model_errors():
    with pytest.raises(RuntimeError, match="model is down"):
        build_full_chain(FakeChatModel(fail=True)).invoke({"input": "hi", "chat_history": []})


# --------------------------------------------------------------------------- memory
def test_get_session_history_creates_once_per_session():
    first = get_session_history("a")
    assert isinstance(first, InMemoryChatMessageHistory)
    assert get_session_history("a") is first
    assert get_session_history("b") is not first
    assert set(chains.store) == {"a", "b"}


def test_bot_saves_each_turn_to_history(fake_llm):
    reply = build_bot(fake_llm).invoke({"input": "Do you ship to Pune?"}, config=session("s1"))
    assert reply == "GENERAL reply"
    messages = get_session_history("s1").messages
    assert [(m.type, m.content) for m in messages] == [
        ("human", "Do you ship to Pune?"), ("ai", "GENERAL reply")]


def test_bot_feeds_earlier_turns_into_the_reply_prompt(fake_llm):
    bot = build_bot(fake_llm)
    bot.invoke({"input": "Do you ship to Pune?"}, config=session("s1"))
    bot.invoke({"input": "And to Delhi?"}, config=session("s1"))
    reply_prompt = fake_llm.calls[-1]
    assert [m.content for m in reply_prompt[1:3]] == ["Do you ship to Pune?", "GENERAL reply"]


def test_bot_keeps_sessions_apart(fake_llm):
    bot = build_bot(fake_llm)
    bot.invoke({"input": "first"}, config=session("s1"))
    bot.invoke({"input": "second"}, config=session("s2"))
    assert len(get_session_history("s1").messages) == 2
    assert len(get_session_history("s2").messages) == 2


def test_bot_streams_the_reply_and_stores_it(fake_llm):
    chunks = list(build_bot(fake_llm).stream({"input": SAMPLE_TICKETS["Safety: button, toddler"]},
                                             config=session("s1")))
    assert len(chunks) > 1, "the reply should arrive in several chunks"
    assert "".join(chunks) == "HUMAN HAND-OFF reply"
    assert get_session_history("s1").messages[-1].content == "HUMAN HAND-OFF reply"


# --------------------------------------------------------------------------- debug + demo helpers
def test_explain_route_reports_the_decision_without_replying(fake_llm):
    info = explain_route(SAMPLE_TICKETS["Value: INR 7,200 sneakers"], fake_llm)
    assert info["route"] == "human_review"
    assert info["needs_human"] is True
    assert info["human_reasons"] == ["order value INR 7,200 is above INR 5,000"]
    assert (info["category"], info["priority"]) == ("genuine_defect", "High")
    assert info["signals"]["order_value"] == 7200
    assert "needs_human=True" in info["triage_summary"]
    assert isinstance(info["latency_s"], float) and info["latency_s"] >= 0
    assert len(fake_llm.calls) == 1, "explain_route runs triage only, no reply"


def test_greeting_chains_pipe_and_sequence_are_equivalent(fake_llm):
    piped, explicit = build_greeting_chains(fake_llm)
    assert piped.invoke({"input": "hi"}) == explicit.invoke({"input": "hi"}) == "DIRECT reply"
    assert fake_llm.calls[0][0].content.startswith("You work for Hopscotch")


def test_full_chain_graph_can_be_drawn(fake_llm):
    drawing = build_full_chain(fake_llm).get_graph().draw_ascii()
    assert "decide_route" in drawing


def test_sample_expectations_cover_every_ticket_and_route():
    assert set(SAMPLE_EXPECTED) == set(SAMPLE_TICKETS)
    assert set(SAMPLE_EXPECTED.values()) == set(chains.ROUTES)
