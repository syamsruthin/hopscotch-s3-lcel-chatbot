"""The deterministic, no-LLM helpers in chains.py: keyword filter, regex extractors, triage parser."""
import pytest

from chains import (
    classify_ticket,
    detect_safety_concern,
    enrich_with_order_id,
    extract_order_id,
    extract_order_value,
    parse_triage,
    quick_filter,
)


# --------------------------------------------------------------------------- classify_ticket
@pytest.mark.parametrize("text, expected", [
    ("I already wore it twice", "likely_policy_violation"),
    ("The return window has EXPIRED", "likely_policy_violation"),
    ("The sole peeled off after one wear.", "likely_genuine_defect"),
    ("The zip is faulty", "likely_genuine_defect"),
    ("Do you ship to Pune?", "needs_llm_classification"),
    # policy words are checked first, so they win over defect words
    ("I worn it and now it's torn", "likely_policy_violation"),
])
def test_classify_ticket(text, expected):
    assert classify_ticket(text) == expected


def test_quick_filter_is_a_runnable_wrapping_classify_ticket():
    texts = ["broke on day one", "hello"]
    assert quick_filter.invoke(texts[0]) == "likely_genuine_defect"
    assert quick_filter.batch(texts) == [classify_ticket(t) for t in texts]


# --------------------------------------------------------------------------- extract_order_id
@pytest.mark.parametrize("source, expected", [
    ("Order #HS88213 arrived late", "HS88213"),
    ("order hs42 please", "HS42"),                     # case-insensitive, normalised to upper
    ("no order id here", None),
    ("HSX123 is not an id", None),
    ({"ticket_text": "about #HS1"}, "HS1"),
    ({"input": "about #HS2"}, "HS2"),
    ({"ticket_text": "", "input": "about #HS3"}, "HS3"),  # empty ticket_text falls back to input
    ({}, None),
])
def test_extract_order_id(source, expected):
    assert extract_order_id(source) == expected


def test_enrich_with_order_id_keeps_existing_keys():
    out = enrich_with_order_id.invoke({"ticket_text": "My order #HS88213 broke", "customer_name": "Ananya"})
    assert out == {"ticket_text": "My order #HS88213 broke", "customer_name": "Ananya", "order_id": "HS88213"}


# --------------------------------------------------------------------------- extract_order_value
@pytest.mark.parametrize("text, expected", [
    ("My order was ₹7,200", 7200),
    ("Paid Rs. 6500 for it", 6500),
    ("paid rs 12000", 12000),
    ("INR 5,500 order", 5500),
    ("₹999 and later ₹2,499", 2499),        # the largest amount wins
    ("It cost 7200", None),                  # no currency marker -> ignored
    ("hers 500 and theirs 900", None),       # 'rs' inside a word is not rupees
    ("", None),
    (None, None),
])
def test_extract_order_value(text, expected):
    assert extract_order_value(text) == expected


# --------------------------------------------------------------------------- detect_safety_concern
@pytest.mark.parametrize("text, reason_fragment", [
    ("He almost swallowed the button", "choking hazard"),
    ("It's a choking risk", "choking hazard"),
    ("Small parts came out of the toy", "choking hazard"),
    ("The buckle injured her", "choking hazard"),
    ("The zip cut my son", "injury"),
    ("She got a rash from the fabric", "skin reaction"),
    ("I think he is allergic to the dye", "skin reaction"),
    ("His skin is itchy and swollen", "skin reaction"),
    ("A bead came loose on my daughter's dress", "small part came loose"),
    ("The bow fell off the baby's shoe", "small part came loose"),
])
def test_safety_concern_flagged(text, reason_fragment):
    result = detect_safety_concern(text)
    assert result["flag"] is True
    assert reason_fragment in result["reason"]


@pytest.mark.parametrize("text", [
    "The sole is peeling",
    "A button came off my jacket",          # 'came off' without a child is not a safety issue
    "The car crash was on the news",        # 'rash' must be a whole word start
    "I'll switch to a different size",      # 'itch' inside 'switch'
    "",
    None,
])
def test_safety_concern_not_flagged(text):
    assert detect_safety_concern(text) == {"flag": False, "reason": None}


def test_strong_safety_pattern_wins_over_weak_one():
    result = detect_safety_concern("A bead came loose and my toddler choked on it")
    assert result["reason"] == "possible child-safety concern (choking hazard / small parts / injury)"


# --------------------------------------------------------------------------- parse_triage
@pytest.mark.parametrize("raw, expected", [
    ("genuine_defect,High", ("genuine_defect", "High")),
    ("policy_violation,low", ("policy_violation", "Low")),
    ("  Other , MEDIUM \n", ("other", "Medium")),
    ("Genuine-Defect,HIGH", ("genuine_defect", "High")),
    ("genuine defect, high", ("genuine_defect", "High")),
    ("policy violation,low", ("policy_violation", "Low")),
    ("category: Genuine_Defect, priority: high", ("genuine_defect", "High")),
    ('"unclear,Medium"', ("unclear", "Medium")),
])
def test_parse_triage_tolerates_messy_output(raw, expected):
    assert parse_triage(raw) == {"category": expected[0], "priority": expected[1]}


@pytest.mark.parametrize("raw", ["", None, "I am not sure", "banana,urgent"])
def test_parse_triage_falls_back_to_safe_default(raw):
    assert parse_triage(raw) == {"category": "unclear", "priority": "Medium"}


def test_parse_triage_ignores_priority_words_inside_other_words():
    # 'highly' must not be read as priority 'high'
    assert parse_triage("other, highly unusual")["priority"] == "Medium"
