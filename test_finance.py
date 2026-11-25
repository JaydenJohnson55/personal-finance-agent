from datetime import date

import finance
from agent import answer, route

TXNS = finance.load("bank_export.csv")


def t(d, desc, amt):
    return {"date": date.fromisoformat(d), "description": desc, "amount": amt}


def test_categories():
    cats = {x["description"]: x["category"] for x in TXNS}
    assert cats["NETFLIX.COM"] == "Subscriptions"
    assert cats["RENT PAYMENT ZELLE"] == "Rent"
    assert cats["PAYROLL DEP CAMPUS JOB"] == "Income"
    assert cats["VENMO PAYMENT J SMITH"] == "Uncategorized"


def test_unknowns_go_to_llm_hook():
    txns = finance.categorize([t("2026-01-01", "MYSTERY SHOP 22", -10.0)], llm=lambda d: "Shopping")
    assert txns[0]["category"] == "Shopping"


def test_merchant_key_ignores_store_numbers():
    assert finance.merchant_key("CHIPOTLE 1123") == finance.merchant_key("CHIPOTLE 2210")
    assert finance.merchant_key("UBER *TRIP") == finance.merchant_key("UBER *TRIP 99")


def test_subscriptions_and_price_increase():
    subs = {s["merchant"]: s for s in finance.find_subscriptions(TXNS)}
    assert "SPOTIFY USA" in subs and subs["SPOTIFY USA"]["kind"] == "subscription"
    assert subs["NETFLIX.COM"]["price_change"] == 2.5
    assert subs["T-MOBILE AUTOPAY"]["kind"] == "bill"
    assert "RENT PAYMENT ZELLE" not in subs


def test_random_repeat_purchases_are_not_subscriptions():
    # Chipotle shows up a lot but not on a monthly schedule
    assert not any("CHIPOTLE" in s["merchant"] for s in finance.find_subscriptions(TXNS))


def test_budget_math_adds_up():
    b = finance.budget_check(TXNS)
    a = b["actual"]
    # each piece is rounded to cents separately, so allow a few cents of drift
    assert abs(a["needs"] + a["wants"] + a["savings"] - b["monthly_income"]) < 0.05
    assert b["target"]["needs"] == round(0.5 * b["monthly_income"], 2)


def test_anomalies():
    flags = finance.find_anomalies(TXNS)
    kinds = {(f["type"], f["description"]) for f in flags}
    assert ("possible duplicate", "CHIPOTLE 1123") in kinds
    assert ("unusually large", "BEST BUY 00123") in kinds


def test_biggest_purchases_skip_recurring_bills():
    big = finance.biggest_purchases(TXNS)
    assert big[0]["description"] == "BEST BUY 00123"
    assert not any("STATE FARM" in b["description"] for b in big)


def test_monthly_summary():
    rows = finance.monthly_summary(TXNS)
    assert [r["month"] for r in rows] == ["2026-08", "2026-09", "2026-10"]
    assert all(r["income"] == 2300 for r in rows)


def test_router_and_offline_answer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert route("what subscriptions do I have?") == "find_subscriptions"
    assert route("was I charged twice anywhere?") == "find_anomalies"
    assert "Netflix".upper() in answer("show my subscriptions", TXNS, use_ai=True)
