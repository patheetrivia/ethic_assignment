import sys
import json
import re
import os
from tabulate import tabulate

from .llm import ask_gpt, extract_ticker
from .tools.company_data import fetch_company_data
from .tools.ranker import rank_from_csv

SP500_CSV = "agent/data/sp500_companies.csv"
TOP_N_DEFAULT = int(os.getenv("TOP_N", "15"))
EXPORT_DIR = os.getenv("EXPORT_DIR", "agent/exports")  # NEW

_LAST_RANKED = None
_LAST_SPEC = None
_LAST_TOPN = None

# Intent check: does the user want a list/ranking? Should eventually
# become smarter / llm-driven
_INTENT_SYSTEM = (
    "You are a classifier. Decide if the user is asking for a ranked/list of companies/stocks.\n"
    "Respond with STRICT JSON only: {\"intent\":\"list|single|other\",\"confidence\":0..1}.\n"
    "Rules:\n"
    "- 'list' if they want top/best/rank/suggest/show multiple companies or a table.\n"
    "- 'single' if they want one ticker/company analyzed.\n"
    "- 'other' otherwise.\n"
    "No prose. No extra keys."
)

# a couple of exemplars help the LLM behave
_INTENT_FEW_SHOTS = [
    ("Show me the top low-volatility green tech stocks", {"intent":"list","confidence":0.95}),
    ("What is AAPL's beta and ESG risk?", {"intent":"single","confidence":0.95}),
    ("Explain dividend yield", {"intent":"other","confidence":0.95}),
]

def _intent_prompt(user_text: str) -> str:
    shots = "\n".join(
        f'User: {u}\nOutput: {json.dumps(o)}'
        for u, o in _INTENT_FEW_SHOTS
    )
    return (
        f"{_INTENT_SYSTEM}\n\n"
        f"{shots}\n\n"
        f"User: {user_text}\nOutput:"
    )

def _safe_parse_json(s: str) -> dict:
    try:
        return json.loads(s.strip())
    except Exception:
        # try to extract a JSON object if the model added stray text
        m = re.search(r"\{.*\}", s, flags=re.S)
        return json.loads(m.group(0)) if m else {}

def is_asking_for_list(text: str) -> bool:
    """LLM-powered intent check with regex fallback."""
    try:
        raw = ask_gpt(_intent_prompt(text))  # uses existing helper
        obj = _safe_parse_json(raw)
        intent = str(obj.get("intent", "")).lower()
        conf = float(obj.get("confidence", 0))
        if intent == "list" and conf >= 0.4:
            return True
        if intent in {"single", "other"} and conf >= 0.6:
            return False
    except Exception:
        pass

# simple affirmative check
_AFFIRM_RE = re.compile(
    r"\b(yes|yep|yeah|y|sure|ok|okay|pls|please|do it|go ahead|sounds good|create it|make it|csv)\b",
    re.IGNORECASE,)

def is_affirmative(text: str) -> bool:
    return bool(_AFFIRM_RE.search(text))

def _criteria_slug(spec: dict, max_len: int = 80) -> str:
    parts = []
    for col, cfg in spec.items():
        d = str(cfg.get("direction", "")).lower()
        if d in {"higher", "high", "positive", "pos"}:
            dshort = "pos"
        elif d in {"lower", "low", "negative", "neg"}:
            dshort = "neg"
        else:
            dshort = "dir"
        parts.append(f"{col}-{dshort}")
    slug = "_".join(parts) or "criteria"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug)  # safe filename
    return slug[:max_len]

def _format_rank_response(ranked_df, spec: dict, top_n: int) -> str:
    # columns to show
    cols = [c for c in ["rank","ticker","name","sector","score_total"] if c in ranked_df.columns]
    view = ranked_df.head(top_n)[cols]
    table = tabulate(view, headers="keys", tablefmt="github", showindex=False)

    # pretty-print criteria from `spec`
    # spec is expected like: {"EnvRisk": {"weight": 0.5, "direction": "lower"}, ...}
    def _arrow(d):
        return {"higher": "↑", "lower": "↓", 1: "↑", -1: "↓"}.get(d, "")

    criteria_lines = []
    for var, cfg in spec.items():
        w = cfg.get("weight", "")
        d = cfg.get("direction", "")
        try:
            w_str = f"{float(w):.2f}"
        except Exception:
            w_str = str(w)
        arrow = _arrow(d)
        # If direction is a numeric 1/-1 or string "higher"/"lower", show a readable phrase
        if d in ("higher", "lower"):
            pref = f"prefers {d}"
        elif d in (1, -1):
            pref = "prefers higher" if d == 1 else "prefers lower"
        else:
            pref = str(d)
        criteria_lines.append(f"- {var}: {pref} {arrow}".rstrip())

    criteria_block = "\n".join(criteria_lines) if criteria_lines else "(no explicit criteria provided)"

    top_count = min(top_n, len(view))
    return (
        f"Here are the top {top_count} companies for your criteria:\n\n"
        f"Criteria:\n{criteria_block}\n\n"
        f"{table}\n\n"
        f"Want a CSV of the Top {top_count}?"
    )

def _export_last_csv() -> str:
    global _LAST_RANKED, _LAST_SPEC, _LAST_TOPN
    if _LAST_RANKED is None or _LAST_SPEC is None or _LAST_TOPN is None:
        return "I don't have a recent ranking to export."

    cols = [c for c in ["rank","ticker","name","sector","score_total"] if c in _LAST_RANKED.columns]
    view = _LAST_RANKED.head(_LAST_TOPN)[cols]

    os.makedirs(EXPORT_DIR, exist_ok=True)
    slug = _criteria_slug(_LAST_SPEC)
    top_count = min(_LAST_TOPN, len(view))
    filename = f"top{top_count}_{slug}.csv"
    out_path = os.path.join(EXPORT_DIR, filename)

    # write exactly the table we displayed
    view.to_csv(out_path, index=False)
    return f"Saved CSV: {out_path}"

_FINANCE_KEYWORDS = re.compile(
    r"\b(stock|stocks|company|companies|ticker|equity|share|market|index|indices|beta|alpha|volatility|esg|risk|finance|financial|return|dividend|portfolio)\b",
    re.I
)

def is_finance_related(text: str) -> bool:
    return bool(_FINANCE_KEYWORDS.search(text))

def handle_input(user_input: str) -> str:
    global _LAST_RANKED, _LAST_SPEC, _LAST_TOPN

    # If the last message offered a CSV and the user now says "yes", export it.
    if is_affirmative(user_input) and _LAST_RANKED is not None:
        return _export_last_csv()

    if is_asking_for_list(user_input):
        ranked, spec = rank_from_csv(
            csv_path=SP500_CSV,
            user_text=user_input,
            top_n=max(TOP_N_DEFAULT, 100),  # fetch enough rows so we can offer a Top-100
            sector_neutral=True,
            out_path=None,
        )

        # store for possible CSV export on user's "yes"
        _LAST_RANKED = ranked
        _LAST_SPEC = spec.get("preferences", spec) if isinstance(spec, dict) else {}
        _LAST_TOPN = TOP_N_DEFAULT

        return _format_rank_response(ranked, spec, top_n=TOP_N_DEFAULT)

    ticker = extract_ticker(user_input)
    if ticker:
        data = fetch_company_data(ticker)   # calls Yahoo

        summary = ask_gpt(
            f"User asked: {user_input}\n\nHere is raw company data: {data}\n\n"
            "Use that data, if possible, to answer the user's question as concisely as possible."
            "If the user is requesting info about sustainability or environmental impact," 
            "consult:  "

        )
        return summary

    # fallback if no ticker found
    if is_finance_related(user_input):
        # still stock/finance-y → let LLM answer
        return ask_gpt(user_input)
    else:
        # clearly unrelated → refuse
        return "That's outside my area of expertise. Have any questions about stocks?"
    return ask_gpt(user_input)

def run():
    print("Finance Agent (type 'exit' to quit).")
    while True:
        try:
            user = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user.strip().lower() in {"exit","quit"}:
            break

        ans = handle_input(user)
        print(ans)

if __name__ == "__main__":
    run()