"""Personal Finance Agent: the analysis tools.

Each function here is a "tool" the agent can call:
  categorize, spending_by_category, find_subscriptions, budget_check,
  find_anomalies, biggest_purchases, monthly_summary
"""

import csv
import re
import statistics
from collections import defaultdict
from datetime import date

CATEGORY_RULES = [
    ("Income", ["payroll", "direct dep", "deposit"]),
    ("Rent", ["rent"]),
    ("Groceries", ["ralphs", "trader joe", "safeway", "vons", "whole foods", "target"]),
    ("Dining", ["chipotle", "starbucks", "in-n-out", "doordash", "taco bell", "mcdonald", "ubereats"]),
    ("Transport", ["uber", "lyft", "metro", "shell", "chevron"]),
    ("Subscriptions", ["spotify", "netflix", "hulu", "chatgpt", "openai", "apple.com/bill", "youtube"]),
    ("Phone & Insurance", ["t-mobile", "verizon", "state farm", "geico"]),
    ("Fitness", ["planet fitness", "gym"]),
    ("Shopping", ["amazon", "nike", "best buy", "steam"]),
]

# 50/30/20 rule: needs / wants / savings
NEEDS = {"Rent", "Groceries", "Transport", "Phone & Insurance"}
WANTS = {"Dining", "Subscriptions", "Fitness", "Shopping", "Uncategorized"}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    txns = []
    for r in rows:
        txns.append({
            "date": date.fromisoformat(r["date"]),
            "description": r["description"].strip(),
            "amount": float(r["amount"]),
        })
    return categorize(txns)


def merchant_key(desc):
    """'CHIPOTLE 1123' and 'CHIPOTLE 2210' are the same merchant: drop store numbers and symbols."""
    d = desc.lower()
    d = re.sub(r"[#*]\S*", " ", d)
    d = re.sub(r"\d+", " ", d)
    d = re.sub(r"\s+", " ", d).strip()
    return d


def categorize(txns, llm=None):
    """Rule-based categories; anything unknown goes to the optional llm(description) function."""
    for t in txns:
        low = t["description"].lower()
        t["category"] = "Uncategorized"
        for cat, keys in CATEGORY_RULES:
            if any(k in low for k in keys):
                t["category"] = cat
                break
        if t["category"] == "Uncategorized" and t["amount"] > 0:
            t["category"] = "Income"
        if t["category"] == "Uncategorized" and llm:
            guess = llm(t["description"])
            if guess:
                t["category"] = guess
    return txns


def months_covered(txns):
    return sorted({(t["date"].year, t["date"].month) for t in txns})


def spending_by_category(txns, month=None):
    out = defaultdict(float)
    for t in txns:
        if t["amount"] < 0 and (month is None or (t["date"].year, t["date"].month) == month):
            out[t["category"]] += -t["amount"]
    return dict(sorted(((k, round(v, 2)) for k, v in out.items()), key=lambda kv: -kv[1]))


def find_subscriptions(txns):
    """Recurring charges: same merchant, roughly monthly, similar amount.

    Also flags price increases, which is how subscriptions quietly cost more.
    """
    by_merchant = defaultdict(list)
    for t in txns:
        if t["amount"] < 0:
            by_merchant[merchant_key(t["description"])].append(t)

    subs = []
    for key, items in by_merchant.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t["date"])
        gaps = [(b["date"] - a["date"]).days for a, b in zip(items, items[1:])]
        amounts = [-t["amount"] for t in items]
        monthly = all(25 <= g <= 35 for g in gaps)
        steady = max(amounts) <= 1.25 * min(amounts)
        if monthly and steady and items[0]["category"] not in ("Rent",):
            latest = amounts[-1]
            sub = {
                "merchant": items[-1]["description"],
                "kind": "bill" if items[-1]["category"] == "Phone & Insurance" else "subscription",
                "category": items[-1]["category"],
                "monthly": round(latest, 2),
                "yearly": round(latest * 12, 2),
                "times_seen": len(items),
                "price_change": round(amounts[-1] - amounts[-2], 2),
            }
            subs.append(sub)
    return sorted(subs, key=lambda s: -s["yearly"])


def monthly_summary(txns):
    rows = []
    for ym in months_covered(txns):
        inc = sum(t["amount"] for t in txns if t["amount"] > 0 and (t["date"].year, t["date"].month) == ym)
        spend = sum(-t["amount"] for t in txns if t["amount"] < 0 and (t["date"].year, t["date"].month) == ym)
        rows.append({"month": f"{ym[0]}-{ym[1]:02d}", "income": round(inc, 2), "spent": round(spend, 2),
                     "saved": round(inc - spend, 2)})
    return rows


def budget_check(txns):
    """Compare actual spending to the 50/30/20 rule, averaged per month."""
    n = len(months_covered(txns)) or 1
    income = sum(t["amount"] for t in txns if t["amount"] > 0) / n
    cats = spending_by_category(txns)
    needs = sum(v for k, v in cats.items() if k in NEEDS) / n
    wants = sum(v for k, v in cats.items() if k in WANTS) / n
    saved = income - needs - wants
    target = {"needs": 0.5 * income, "wants": 0.3 * income, "savings": 0.2 * income}
    actual = {"needs": needs, "wants": wants, "savings": saved}
    return {
        "monthly_income": round(income, 2),
        "actual": {k: round(v, 2) for k, v in actual.items()},
        "target": {k: round(v, 2) for k, v in target.items()},
        "pct_of_income": {k: round(100 * v / income, 1) if income else 0 for k, v in actual.items()},
    }


def find_anomalies(txns, z=2.5):
    """Duplicate charges, plus purchases far above normal for their category."""
    flags = []
    seen = {}
    for t in txns:
        key = (t["date"], t["description"], t["amount"])
        if t["amount"] < 0 and key in seen:
            flags.append({"type": "possible duplicate", "date": t["date"].isoformat(),
                          "description": t["description"], "amount": -t["amount"]})
        seen[key] = True

    by_cat = defaultdict(list)
    for t in txns:
        if t["amount"] < 0:
            by_cat[t["category"]].append(-t["amount"])
    for t in txns:
        vals = by_cat.get(t["category"], [])
        if t["amount"] >= 0 or len(vals) < 4:
            continue
        others = list(vals)
        others.remove(-t["amount"])  # compare to the rest of the category, not including itself
        mean, sd = statistics.mean(others), statistics.pstdev(others)
        if sd and (-t["amount"] - mean) / sd > z:
            flags.append({"type": "unusually large", "date": t["date"].isoformat(),
                          "description": t["description"], "amount": -t["amount"],
                          "typical": round(mean, 2)})
    return flags


def biggest_purchases(txns, n=5):
    """One-off purchases only: rent and recurring bills/subscriptions are left out."""
    recurring = {merchant_key(s["merchant"]) for s in find_subscriptions(txns)}
    spend = [t for t in txns if t["amount"] < 0 and t["category"] not in ("Rent",)
             and merchant_key(t["description"]) not in recurring]
    spend.sort(key=lambda t: t["amount"])
    return [{"date": t["date"].isoformat(), "description": t["description"], "amount": -t["amount"]}
            for t in spend[:n]]
