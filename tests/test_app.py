"""app.py, driven headlessly with Streamlit's AppTest and the FakeChatModel (no network)."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import chains
import llm
from chains import SAMPLE_TICKETS
from conftest import FakeChatModel, sample_triage

APP = str(Path(__file__).parents[1] / "app.py")


@pytest.fixture
def make_app(monkeypatch):
    """Return a factory that runs app.py with the given fake model behind get_llm()."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def _make(model=None):
        model = model or FakeChatModel(triage_reply=sample_triage)
        monkeypatch.setattr(llm, "get_llm", lambda *a, **k: model)
        st.cache_resource.clear()  # get_bot() is cached across runs; rebuild it with this model
        return AppTest.from_file(APP, default_timeout=30).run()

    yield _make
    st.cache_resource.clear()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def texts(elements):
    return [e.value for e in elements]


# --------------------------------------------------------------------------- startup
def test_app_stops_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "OPENROUTER_API_KEY is not set" in at.error[0].value
    assert not at.tabs


def test_app_stops_when_the_model_client_cannot_be_built(make_app, monkeypatch):
    at = make_app()

    def broken(*a, **k):
        raise ValueError("bad config")
    monkeypatch.setattr(llm, "get_llm", broken)
    at.run()
    assert at.error[0].value == "Could not create the model client: bad config"


def test_app_renders_sidebar_and_three_tabs(make_app):
    at = make_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["💬 Chat", "🔗 Pipeline", "🧪 Runnables lab"]
    assert f"Model: {llm.DEFAULT_MODEL}" in texts(at.sidebar.caption)
    assert at.session_state.session_id in at.sidebar.caption[1].value


def test_new_session_button_changes_the_session_id(make_app):
    at = make_app()
    old = at.session_state.session_id
    button(at, "New session").click().run()
    assert at.session_state.session_id != old


# --------------------------------------------------------------------------- chat tab
def test_chat_shows_a_hint_when_history_is_empty(make_app):
    at = make_app()
    assert any("Try:" in c for c in texts(at.tabs[0].caption))


def test_chat_replies_and_stores_the_turn(make_app):
    at = make_app()
    at.chat_input[0].set_value("Do you ship to Pune?").run()
    assert not at.exception
    history = chains.get_session_history(at.session_state.session_id).messages
    assert [m.content for m in history] == ["Do you ship to Pune?", "GENERAL reply"]
    assert any(c.startswith("⏱️") for c in texts(at.caption))

    at.run()  # the next rerun replays the stored history as chat messages
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "GENERAL reply" in texts(at.markdown)


def test_facilitator_mode_shows_the_internal_triage_panel(make_app):
    at = make_app()
    at.toggle[0].set_value(True).run()
    at.chat_input[0].set_value(SAMPLE_TICKETS["Value: INR 7,200 sneakers"]).run()
    assert not at.exception
    assert at.expander[0].label == "🛠️ Internal triage - 🚨 Human review"
    assert "- order value INR 7,200 is above INR 5,000" in texts(at.markdown)
    assert len(at.table) == 1

    at.run()  # the panel is kept in debug_log and re-rendered under the stored reply
    assert any(e.label.startswith("🛠️ Internal triage") for e in at.expander)


def test_facilitator_panel_without_human_reasons(make_app):
    at = make_app()
    at.toggle[0].set_value(True).run()
    at.chat_input[0].set_value("Do you ship to Pune?").run()
    assert at.expander[0].label == "🛠️ Internal triage - 💬 General"
    assert "**Why a human:**" not in texts(at.markdown)


def test_chat_shows_an_error_when_the_model_fails(make_app):
    at = make_app(FakeChatModel(fail=True))
    at.chat_input[0].set_value("hello").run()
    assert "Something went wrong calling the model: model is down" in texts(at.error)


# --------------------------------------------------------------------------- pipeline tab
def test_pipeline_tab_shows_source_graph_and_prompts(make_app):
    at = make_app()
    code = "\n".join(texts(at.tabs[1].code))
    assert "def build_full_chain" in code
    assert "def build_route_branch" in code
    assert "decide_route" in code              # from the draw_ascii() graph
    assert "[ROUTE: HUMAN HAND-OFF]" in code    # branch prompts, policy text stripped
    assert "--- POLICY ---" not in code


def test_pipeline_tab_reports_when_the_graph_cannot_be_drawn(make_app, monkeypatch):
    at = make_app()

    def broken(_llm):
        raise RuntimeError("no graph")
    monkeypatch.setattr(chains, "build_full_chain", broken)
    at.run()
    assert "Could not draw the graph: no graph" in texts(at.info)


@pytest.mark.parametrize("sample, route_badge, reply", [
    ("Defect: Dino-Print sneakers", "🧵 Defect", "GENUINE DEFECT reply"),
    ("Safety: button, toddler", "🚨 Human review", "HUMAN HAND-OFF reply"),
])
def test_pipeline_step_by_step(make_app, sample, route_badge, reply):
    at = make_app()
    at.selectbox(key="pipe_sample").set_value(sample).run()
    assert at.text_area(key=f"pipe_text_{sample}").value == SAMPLE_TICKETS[sample]
    button(at, "Run step-by-step").click().run()
    assert not at.exception
    assert f"**Stage 3 - `RunnableBranch` chose:** {route_badge}" in texts(at.markdown)
    assert texts(at.info)[-1] == reply
    assert len(at.tabs[1].json) == 3  # input, signals, decision


def test_pipeline_step_by_step_shows_an_error_when_the_model_fails(make_app):
    at = make_app(FakeChatModel(fail=True))
    button(at, "Run step-by-step").click().run()
    assert "Pipeline failed: model is down" in texts(at.error)


# --------------------------------------------------------------------------- runnables lab tab
def test_lab_pipe_vs_sequence(make_app):
    at = make_app()
    at.button(key="btn_a").click().run()
    assert texts(at.tabs[2].markdown).count("DIRECT reply") == 2


def test_lab_quick_filter_and_assign_need_no_model(make_app):
    at = make_app()
    at.text_input(key="lab_b").set_value("I already wore it").run()
    assert "quick_filter.invoke('I already wore it')  ->  'likely_policy_violation'" in texts(at.code)
    assert at.json[0].value == ('{"ticket_text": "My order #HS88213 arrived with a broken zipper.", '
                                '"customer_name": "Ananya", "order_id": "HS88213"}')


def test_lab_batch_triage_matches_expected_routes(make_app):
    at = make_app()
    at.button(key="btn_d").click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["ticket"]) == list(SAMPLE_TICKETS)
    assert list(table["route"]) == list(table["expected"])
    assert any(c.startswith("6 tickets in") for c in texts(at.caption))


def test_lab_stream(make_app):
    at = make_app()
    at.button(key="btn_e").click().run()
    assert "DIRECT reply" in texts(at.tabs[2].markdown)


def test_lab_sequential_vs_parallel(make_app):
    at = make_app()
    at.button(key="btn_f").click().run()
    assert not at.exception
    sequential, parallel = at.json[-2].value, at.json[-1].value
    assert '"order_id": "HS88213"' in sequential and '"order_id": "HS88213"' in parallel
    assert any(c.startswith("Same signals either way") for c in texts(at.caption))


@pytest.mark.parametrize("key", ["btn_a", "btn_d", "btn_e", "btn_f"])
def test_lab_buttons_show_an_error_when_the_model_fails(make_app, key):
    at = make_app(FakeChatModel(fail=True))
    at.button(key=key).click().run()
    assert "model is down" in texts(at.error)
