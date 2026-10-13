"""decide_route: policy Section 4 (mandatory human review) enforced in plain code."""
import pytest

from chains import ORDER_VALUE_LIMIT, ROUTES, decide_route


def make_input(category="other", priority="Low", safety=None, order_value=None, order_id=None):
    return {
        "input": "customer text",
        "chat_history": [],
        "signals": {
            "triage": {"category": category, "priority": priority},
            "safety": safety or {"flag": False, "reason": None},
            "order_value": order_value,
            "order_id": order_id,
        },
    }


@pytest.mark.parametrize("category, route", [
    ("genuine_defect", "defect"),
    ("policy_violation", "policy"),
    ("other", "general"),
])
def test_clean_cases_follow_the_triage_category(category, route):
    out = decide_route(make_input(category=category))
    assert out["route"] == route
    assert out["needs_human"] is False
    assert out["human_reasons"] == []


def test_unclear_goes_to_a_human():
    out = decide_route(make_input(category="unclear", priority="Medium"))
    assert out["route"] == "human_review"
    assert out["needs_human"] is True
    assert out["human_reasons"] == ["facts are unclear, missing or conflicting (possible defect vs misuse)"]


def test_safety_concern_overrides_triage_and_forces_high_priority():
    safety = {"flag": True, "reason": "skin reaction or allergy complaint"}
    out = decide_route(make_input(category="other", priority="Low", safety=safety))
    assert out["route"] == "human_review"
    assert out["priority"] == "High"
    assert out["category"] == "other", "the triage category is reported unchanged"
    assert out["human_reasons"] == ["skin reaction or allergy complaint"]


def test_safety_flag_without_reason_gets_a_generic_reason():
    out = decide_route(make_input(safety={"flag": True, "reason": None}))
    assert out["human_reasons"] == ["safety concern"]


def test_order_value_at_the_limit_is_not_escalated():
    out = decide_route(make_input(category="genuine_defect", order_value=ORDER_VALUE_LIMIT))
    assert out["route"] == "defect"


def test_order_value_above_the_limit_is_escalated():
    out = decide_route(make_input(category="genuine_defect", order_value=ORDER_VALUE_LIMIT + 1))
    assert out["route"] == "human_review"
    assert out["human_reasons"] == ["order value INR 5,001 is above INR 5,000"]


def test_all_reasons_are_collected_in_order():
    safety = {"flag": True, "reason": "possible child-safety concern (injury)"}
    out = decide_route(make_input(category="unclear", safety=safety, order_value=9000))
    assert out["human_reasons"] == [
        "possible child-safety concern (injury)",
        "order value INR 9,000 is above INR 5,000",
        "facts are unclear, missing or conflicting (possible defect vs misuse)",
    ]


@pytest.mark.parametrize("signals", [
    {"triage": None, "safety": None, "order_value": None},
    {},
])
def test_missing_signals_default_to_unclear_and_a_human(signals):
    out = decide_route({"input": "x", "signals": signals})
    assert (out["category"], out["priority"], out["route"]) == ("unclear", "Medium", "human_review")


def test_output_keeps_the_input_keys():
    x = make_input()
    out = decide_route(x)
    for key in ("input", "chat_history", "signals"):
        assert out[key] is x[key]
    assert out["route"] in ROUTES


def test_triage_summary_without_extras():
    out = decide_route(make_input(category="genuine_defect", priority="High"))
    assert out["triage_summary"] == "category=genuine_defect; priority=High; needs_human=False"


def test_triage_summary_lists_reasons_order_id_and_value():
    out = decide_route(make_input(category="genuine_defect", priority="High",
                                  order_value=7200, order_id="HS88213"))
    assert out["triage_summary"] == (
        "category=genuine_defect; priority=High; needs_human=True"
        "; reasons=order value INR 7,200 is above INR 5,000"
        "; order_id=HS88213; order_value=INR 7200"
    )
