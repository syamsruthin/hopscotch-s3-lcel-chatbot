"""Hopscotch Support (Session 3) - LCEL pipeline with RunnableParallel + RunnableBranch."""
import inspect
import os
import time
import uuid

import pandas as pd
import streamlit as st

import chains
from chains import (
    SAMPLE_EXPECTED, SAMPLE_TICKETS, build_bot, build_enrichment_pipeline, build_full_chain,
    build_greeting_chains, build_route_branch, build_signals, decide_route, enrich_with_order_id, explain_route,
    quick_filter,
)
from llm import DEFAULT_MODEL, get_llm
from prompts import DEFECT_PROMPT, ESCALATION_PROMPT, GENERAL_PROMPT, POLICY_REPLY_PROMPT

st.set_page_config(page_title="Hopscotch Support (Session 3)", page_icon="🧵", layout="wide")

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key "
             "(or set it in the Streamlit Cloud app Secrets).")
    st.stop()

ROUTE_BADGES = {
    "human_review": "🚨 Human review",
    "defect": "🧵 Defect",
    "policy": "📋 Policy",
    "general": "💬 General",
}


@st.cache_resource
def get_store() -> dict:
    """Chat histories live here so they survive Streamlit reruns (same dict as chains.store)."""
    return chains.store


@st.cache_resource
def get_bot():
    """One bot for the whole app; get_session_history reads the shared store."""
    return build_bot(get_llm())


store = get_store()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🧵 Hopscotch")
    st.caption(f"Model: {DEFAULT_MODEL}")
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    st.caption(f"Session id: `{st.session_state.session_id}`")
    if st.button("New session"):
        st.session_state.session_id = uuid.uuid4().hex[:8]
        st.rerun()
    facilitator = st.toggle("🛠️ Facilitator mode: show internal triage", value=False,
                            help="Adds one extra LLM call per turn to show the routing decision.")

sid = st.session_state.session_id
try:
    llm = get_llm()
except Exception as e:
    st.error(f"Could not create the model client: {e}")
    st.stop()
st.session_state.setdefault("debug_log", {})  # (session, message index) -> explain_route output


def render_debug(info: dict):
    """Facilitator panel: route badge, reasons, the 4 parallel signals, latency."""
    with st.expander(f"🛠️ Internal triage - {ROUTE_BADGES[info['route']]}", expanded=False):
        st.markdown(f"**Route:** {ROUTE_BADGES[info['route']]}  |  **Latency (signals):** {info['latency_s']}s")
        if info["human_reasons"]:
            st.markdown("**Why a human:**")
            for r in info["human_reasons"]:
                st.markdown(f"- {r}")
        s = info["signals"]
        st.table(pd.DataFrame([
            {"signal": "triage (LLM)", "value": f"{s['triage']['category']} / {s['triage']['priority']}"},
            {"signal": "safety (keywords)", "value": str(s["safety"])},
            {"signal": "order_value (regex)", "value": str(s["order_value"])},
            {"signal": "order_id (regex)", "value": str(s["order_id"])},
        ]))


tab_chat, tab_pipe, tab_lab = st.tabs(["💬 Chat", "🔗 Pipeline", "🧪 Runnables lab"])

# ------------------------------------------------------------------ tab 1: chat
with tab_chat:
    st.subheader("Hopscotch Support")
    history = chains.get_session_history(sid).messages
    if not history:
        st.caption("Try: 'The sole of my son's sneakers is peeling after 3 wears' or "
                   "'The button came off my toddler's shirt'.")
    for i, m in enumerate(history):
        with st.chat_message("user" if m.type == "human" else "assistant"):
            st.markdown(m.content)
            if facilitator and (sid, i) in st.session_state.debug_log:
                render_debug(st.session_state.debug_log[(sid, i)])

    msg = st.chat_input("How can we help?")
    if msg:
        with st.chat_message("user"):
            st.markdown(msg)
        with st.chat_message("assistant"):
            try:
                t0 = time.perf_counter()
                bot = get_bot()
                # .stream() yields string chunks; RunnableWithMessageHistory saves the
                # complete reply to the store once the stream finishes.
                st.write_stream(bot.stream({"input": msg}, config={"configurable": {"session_id": sid}}))
                st.caption(f"⏱️ {time.perf_counter() - t0:.1f}s total")
                if facilitator:
                    info = explain_route(msg, llm)  # one extra LLM call, debug only
                    idx = len(chains.get_session_history(sid).messages) - 1
                    st.session_state.debug_log[(sid, idx)] = info
                    render_debug(info)
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong calling the model: {e}")

# ------------------------------------------------------------------ tab 2: pipeline
DOT = """
digraph G {
  rankdir=TB; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#7c6ae6", fillcolor="#efeafe", fontcolor="#222"];
  edge [color="#888"];
  input [label="input + chat_history", fillcolor="#e6f7f5", color="#1fa89a"];
  assign [label="RunnablePassthrough.assign(signals = ...)"];
  subgraph cluster_par { label="RunnableParallel (concurrent)"; fontname="Helvetica"; fontsize=10; color="#aaa"; style="dashed";
    triage [label="triage\\n(LLM)"]; safety [label="safety\\n(keywords)"]; value [label="order_value\\n(regex)"]; oid [label="order_id\\n(regex)"]; }
  decide [label="RunnableLambda(decide_route)\\nneeds_human, route, triage_summary"];
  subgraph cluster_br { label="RunnableBranch"; fontname="Helvetica"; fontsize=10; color="#aaa"; style="dashed";
    human [label="human_review\\nESCALATION_PROMPT | llm", fillcolor="#fde8e8", color="#d9534f"];
    defect [label="defect\\nDEFECT_PROMPT | llm"]; policy [label="policy\\nPOLICY_REPLY_PROMPT | llm"]; general [label="general (default)\\nGENERAL_PROMPT | llm"]; }
  parser [label="StrOutputParser"];
  reply [label="reply (streamed)", fillcolor="#e6f7f5", color="#1fa89a"];
  input -> assign; assign -> triage; assign -> safety; assign -> value; assign -> oid;
  triage -> decide; safety -> decide; value -> decide; oid -> decide;
  decide -> human; decide -> defect; decide -> policy; decide -> general;
  human -> parser; defect -> parser; policy -> parser; general -> parser; parser -> reply;
}
"""

with tab_pipe:
    st.subheader("The LCEL pipeline")
    st.graphviz_chart(DOT, width="stretch")
    st.markdown("**`build_full_chain` (from `chains.py`)**")
    st.code(inspect.getsource(build_full_chain), language="python")
    st.code(inspect.getsource(build_route_branch), language="python")
    with st.expander("LangChain's own graph: full_chain.get_graph().draw_ascii()"):
        try:
            st.code(build_full_chain(llm).get_graph().draw_ascii(), language="text")
        except Exception as e:  # noqa: BLE001
            st.info(f"Could not draw the graph: {e}")

    with st.expander("Branch prompts: the system prompt each RunnableBranch route uses"):
        for route, branch_prompt in [("human_review", ESCALATION_PROMPT), ("defect", DEFECT_PROMPT),
                                     ("policy", POLICY_REPLY_PROMPT), ("general (default)", GENERAL_PROMPT)]:
            st.markdown(f"**{route}**")
            # messages[0] is the system message; {policy} is filled in by .partial(), so show the rules only
            st.code(branch_prompt.messages[0].prompt.template.split("\n\n--- POLICY ---")[0], language="text")

    st.markdown("### Run step-by-step")
    key = st.selectbox("Sample ticket", list(SAMPLE_TICKETS), key="pipe_sample")
    text = st.text_area("Customer message", SAMPLE_TICKETS[key], key=f"pipe_text_{key}")
    if st.button("Run step-by-step"):
        try:
            payload = {"input": text, "chat_history": []}
            st.markdown("**Stage 0 - input**")
            st.json(payload)
            with st.spinner("Stage 1: RunnableParallel signals..."):
                signals = build_signals(llm).invoke(text)
            st.markdown("**Stage 1 - `RunnablePassthrough.assign(signals=...)`** (adds the `signals` key)")
            st.json(signals)
            routed = decide_route({**payload, "signals": signals})
            st.markdown("**Stage 2 - `decide_route`**")
            st.json({k: routed[k] for k in ("needs_human", "human_reasons", "category", "priority",
                                            "route", "triage_summary")})
            st.markdown(f"**Stage 3 - `RunnableBranch` chose:** {ROUTE_BADGES[routed['route']]}")
            with st.spinner("Stage 4: reply prompt | llm | parser..."):
                reply = build_route_branch(llm).invoke(routed)
            st.markdown("**Stage 4 - `StrOutputParser` output (the reply)**")
            st.info(reply)
        except Exception as e:  # noqa: BLE001
            st.error(f"Pipeline failed: {e}")

# ------------------------------------------------------------------ tab 3: lab
with tab_lab:
    st.subheader("Runnables lab")

    st.markdown("#### (a) `|` pipe vs explicit `RunnableSequence`")
    c1, c2 = st.columns(2)
    c1.code("chain = prompt | llm | StrOutputParser()", language="python")
    c2.code("chain = RunnableSequence(\n    first=prompt,\n    middle=[llm],\n    last=StrOutputParser(),\n)",
            language="python")
    q = st.text_input("Question", "Do you have raincoats for toddlers?", key="lab_a")
    if st.button("Run both", key="btn_a"):
        try:
            piped, explicit = build_greeting_chains(llm)
            c1.write(piped.invoke({"input": q}))
            c2.write(explicit.invoke({"input": q}))
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.markdown("#### (b) `quick_filter.invoke(...)` - a RunnableLambda, no LLM")
    t_b = st.text_input("Ticket text", "The sole peeled off after one wear.", key="lab_b")
    st.code(f"quick_filter.invoke({t_b!r})  ->  {quick_filter.invoke(t_b)!r}", language="python")

    st.markdown("#### (c) `RunnablePassthrough.assign` keeps the original keys")
    t_c = st.text_input("Ticket text", "My order #HS88213 arrived with a broken zipper.", key="lab_c")
    st.json(enrich_with_order_id.invoke({"ticket_text": t_c, "customer_name": "Ananya"}))

    st.markdown("#### (d) `.batch()` over all sample tickets")
    if st.button("Triage all samples (batch)", key="btn_d"):
        try:
            names, texts = list(SAMPLE_TICKETS), list(SAMPLE_TICKETS.values())
            t0 = time.perf_counter()
            all_signals = build_signals(llm).batch(texts, config={"max_concurrency": 3})
            took = time.perf_counter() - t0
            rows = []
            for n, tx, s in zip(names, texts, all_signals):
                d = decide_route({"input": tx, "signals": s})
                rows.append({"ticket": n, "category": d["category"], "priority": d["priority"],
                             "safety": s["safety"]["flag"], "order_value": s["order_value"],
                             "route": ROUTE_BADGES[d["route"]],
                             "expected": ROUTE_BADGES[SAMPLE_EXPECTED[n]]})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(f"{len(texts)} tickets in {took:.1f}s with max_concurrency=3")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.markdown("#### (e) `.stream()` token by token")
    q_e = st.text_input("Question", "Tell me about your return policy.", key="lab_e")
    if st.button("Stream", key="btn_e"):
        try:
            piped, _ = build_greeting_chains(llm)
            st.write_stream(piped.stream({"input": q_e}))
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.markdown("#### (f) `.assign().assign()` vs `RunnableParallel`")
    st.code("""# sequential: each .assign() waits for the previous one
build_enrichment_pipeline = (
    RunnablePassthrough.assign(triage=text | triage_subchain)   # LLM call first...
    .assign(safety=...).assign(order_value=...).assign(order_id=...)  # ...then the cheap checks
)
# concurrent: all four start together
build_signals = RunnableParallel(triage=..., safety=..., order_value=..., order_id=...)""", language="python")
    t_f = st.text_area("Ticket", SAMPLE_TICKETS["Defect: Dino-Print sneakers"], key="lab_f")
    if st.button("Run both and compare", key="btn_f"):
        try:
            t0 = time.perf_counter()
            seq_out = build_enrichment_pipeline(llm).invoke({"input": t_f})
            seq_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            par_out = build_signals(llm).invoke(t_f)
            par_s = time.perf_counter() - t0
            f1, f2 = st.columns(2)
            f1.markdown(f"**`.assign()` sequence** - {seq_s:.2f}s")
            f1.json(seq_out)
            f2.markdown(f"**`RunnableParallel`** - {par_s:.2f}s")
            f2.json(par_out)
            st.caption("Same signals either way. The difference: the assign steps run one at a time, "
                       "RunnableParallel runs its steps concurrently. Only one step here is a slow LLM "
                       "call, so the gap is small - with several LLM steps it would be large.")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))