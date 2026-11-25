"""Personal Finance Agent: ask questions about your spending in plain English.

  python agent.py bank_export.csv              (chat)
  python agent.py bank_export.csv --report      (one-shot report + chart)

Two modes:
  * demo (default): a simple intent router picks the tool, works offline
  * ai: Claude decides which tools to call (tool use) and explains the result
        (needs ANTHROPIC_API_KEY and `pip install anthropic`)

Privacy: in AI mode only tool results (totals, merchant names) are sent to the
model, never the raw CSV. In demo mode nothing leaves your computer.
"""

import argparse
import json
import os
import sys

import finance

TOOLS = {
    "spending_by_category": "Total spending per category (optionally for one month 'YYYY-MM').",
    "find_subscriptions": "Recurring monthly charges, yearly cost, and price increases.",
    "budget_check": "Compare spending to the 50/30/20 budget rule.",
    "find_anomalies": "Possible duplicate charges and unusually large purchases.",
    "biggest_purchases": "The largest purchases (rent excluded).",
    "monthly_summary": "Income, spending and savings per month.",
}

INTENTS = [
    ("find_subscriptions", ["subscription", "recurring", "netflix", "spotify", "cancel", "monthly charge"]),
    ("find_anomalies", ["weird", "duplicate", "fraud", "unusual", "anomal", "charged twice"]),
    ("budget_check", ["budget", "50/30/20", "saving enough", "on track", "afford"]),
    ("biggest_purchases", ["biggest", "largest", "most expensive"]),
    ("monthly_summary", ["month", "save", "saving", "income", "overview"]),
    ("spending_by_category", ["spend", "spent", "category", "food", "dining", "where"]),
]


def run_tool(name, txns, args=None):
    args = args or {}
    if name == "spending_by_category":
        month = args.get("month")
        if month:
            y, m = month.split("-")
            return finance.spending_by_category(txns, (int(y), int(m)))
        return finance.spending_by_category(txns)
    return getattr(finance, name)(txns)


def route(question):
    q = question.lower()
    for tool, keys in INTENTS:
        if any(k in q for k in keys):
            return tool
    return "monthly_summary"


def explain(tool, result):
    """Plain-English answer for demo mode."""
    if tool == "find_subscriptions":
        subs = [s for s in result if s["kind"] == "subscription"]
        bills = [s for s in result if s["kind"] == "bill"]
        total = sum(s["yearly"] for s in subs)
        lines = [f"You have {len(subs)} subscriptions costing about ${total:,.0f} a year:"]
        for s in subs:
            note = f" (price went up ${s['price_change']:.2f})" if s["price_change"] > 0 else ""
            lines.append(f"  - {s['merchant']}: ${s['monthly']:.2f}/mo, ${s['yearly']:.0f}/yr{note}")
        if bills:
            lines.append("Recurring bills (not subscriptions): " +
                         ", ".join(f"{b['merchant']} ${b['monthly']:.2f}/mo" for b in bills))
        return "\n".join(lines)
    if tool == "budget_check":
        a, t, p = result["actual"], result["target"], result["pct_of_income"]
        return (f"Monthly income ~${result['monthly_income']:,.0f}.\n"
                f"  Needs:   ${a['needs']:,.0f} ({p['needs']}%), target 50% = ${t['needs']:,.0f}\n"
                f"  Wants:   ${a['wants']:,.0f} ({p['wants']}%), target 30% = ${t['wants']:,.0f}\n"
                f"  Savings: ${a['savings']:,.0f} ({p['savings']}%), target 20% = ${t['savings']:,.0f}")
    if tool == "find_anomalies":
        if not result:
            return "Nothing unusual found."
        return "\n".join(f"  - {r['type']}: {r['date']} {r['description']} ${r['amount']:.2f}"
                         + (f" (you usually spend ~${r['typical']:.0f})" if "typical" in r else "")
                         for r in result)
    if tool == "biggest_purchases":
        return "\n".join(f"  - {r['date']} {r['description']} ${r['amount']:.2f}" for r in result)
    if tool == "monthly_summary":
        return "\n".join(f"  {r['month']}: earned ${r['income']:,.0f}, spent ${r['spent']:,.0f}, "
                         f"saved ${r['saved']:,.0f}" for r in result)
    return "\n".join(f"  {k}: ${v:,.2f}" for k, v in result.items())


def ask_claude(question, txns):
    """Claude tool use: the model picks tools, we run them locally, it writes the answer."""
    import anthropic
    client = anthropic.Anthropic()
    tools = [{"name": n, "description": d, "input_schema": {
        "type": "object", "properties": {"month": {"type": "string"}} if n == "spending_by_category" else {}}}
        for n, d in TOOLS.items()]
    messages = [{"role": "user", "content": question}]
    system = ("You are a friendly personal finance assistant for a college student. Use the tools to answer "
              "with real numbers. Be specific and practical. You are not a financial advisor; say so if asked "
              "for investment advice.")
    for _ in range(5):
        msg = client.messages.create(model=os.environ.get("FINANCE_MODEL", "claude-sonnet-4-5"),
                                     max_tokens=1000, system=system, tools=tools, messages=messages)
        if msg.stop_reason != "tool_use":
            return "".join(b.text for b in msg.content if b.type == "text")
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for b in msg.content:
            if b.type == "tool_use":
                out = run_tool(b.name, txns, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    return "Sorry, I couldn't finish that one."


def answer(question, txns, use_ai=False):
    if use_ai and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ask_claude(question, txns)
        except ImportError:
            pass
    tool = route(question)
    return explain(tool, run_tool(tool, txns))


def write_chart(txns, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = finance.months_covered(txns)
    cats = [c for c in finance.spending_by_category(txns) if c != "Rent"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottoms = [0] * len(months)
    for c in cats:
        vals = [finance.spending_by_category(txns, m).get(c, 0) for m in months]
        ax.bar([f"{y}-{m:02d}" for y, m in months], vals, bottom=bottoms, label=c)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Spent ($), rent excluded")
    ax.set_title("Spending by category per month")
    ax.set_ylim(0, max(bottoms) * 1.08)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--ai", action="store_true")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    txns = finance.load(args.csv)

    if args.report:
        for title, q in [("Monthly overview", "overview"), ("Budget (50/30/20)", "budget"),
                         ("Subscriptions", "subscriptions"), ("Anything weird?", "anything unusual"),
                         ("Biggest purchases", "biggest purchases")]:
            print(f"== {title} ==\n{answer(q, txns, args.ai)}\n")
        write_chart(txns, "spending.png")
        print("Chart saved to spending.png")
        return

    print("Ask about your money (blank line to quit). Try: 'what subscriptions do I have?'")
    for line in sys.stdin:
        if not line.strip():
            break
        print(answer(line.strip(), txns, args.ai), "\n")


if __name__ == "__main__":
    main()
