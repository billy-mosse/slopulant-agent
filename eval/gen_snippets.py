"""One-off generator for eval/snippets/gen/: LLM-written rewrites of every system on
main (positives) and unrelated systems (negatives). Output is committed so the eval
set stays fixed; rerun only to grow it.

    python -m eval.gen_snippets
"""
import ast
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from eval.build_cases import similarity_truth
from eval.prompt_lab import COMPANY_REPO, folder_code, git
from watcher import cards, config

OUT = Path(__file__).parent / "snippets" / "gen"

STYLES = {
    "terse": "functional and terse: short variable names, no docstrings or comments, different function names",
    "verbose": "class-based with long descriptive names and a docstring written from the product team's "
               "point of view (no table names), different structure",
    "alt": "a different but equivalent technique or library for the same job (e.g. another similarity "
           "measure, another model family), different names and structure",
}

REWRITE_PROMPT = """Rewrite the Python module below so it does the same job but looks like it was
written independently by another engineer at the same company. Style: {style}.
Do not reuse its function names, variable names or docstring wording. 15-50 lines.
Output only the Python code, no markdown fences, no explanation.

{code}"""

NEGATIVE_TASKS = {
    "ticket_routing": "route customer-support tickets to the right team by classifying the ticket text "
                      "(keyword rules + naive Bayes)",
    "route_planner": "order delivery stops for a van to minimise driving distance (nearest-neighbour heuristic)",
    "coupon_abuse": "flag accounts that abuse sign-up coupons (shared devices, addresses, payment cards)",
    "pick_path": "order warehouse bin visits for a picker to minimise walking (aisle serpentine order)",
    "payment_retry": "schedule retries for failed card payments with exponential backoff and a daily cap",
    "carbon_estimate": "estimate CO2 per shipment from distance, weight and carrier mode",
    "budget_bandit": "allocate daily marketing budget across channels with a Thompson-sampling bandit",
    "supplier_leadtime": "estimate supplier lead times per vendor from purchase-order history (median, p90)",
    "store_traffic_anomaly": "detect anomalous hourly foot traffic in physical stores with a rolling z-score",
    "giftcard_recon": "reconcile gift-card balances between the ledger and the payment processor",
    "seq_ab_monitor": "monitor a running A/B test on checkout conversion with a sequential probability ratio test",
    "label_printer": "generate shipping label payloads (address formatting, barcode string, carrier service code)",
}

# Generated "unrelated" systems that turned out to overlap systems added later.
NEGATIVE_LABELS = {
    "seq_ab_monitor": {"duplicate": ["ab_test_analysis"], "related": []},
    "ticket_routing": {"duplicate": [], "related": ["returns_reason_classifier"]},
    "supplier_leadtime": {"duplicate": [], "related": ["replenishment"]},
    "coupon_abuse": {"duplicate": [], "related": ["fake_review_detection"]},
    "budget_bandit": {"duplicate": [], "related": ["next_best_offer"]},
    "store_traffic_anomaly": {"duplicate": [], "related": ["model_monitoring"]},
}

NEGATIVE_PROMPT = """Write a Python module (15-50 lines) for the ML/data team of a home-goods
e-commerce company. It should: {task}.
Start with a short docstring including Input: and Output: lines naming tables as schema.table.
Output only the Python code, no markdown fences, no explanation."""


def clean(text):
    text = re.sub(r"^```\w*\n|\n```\s*$", "", text.strip())
    text = re.sub(r"(?m)^```\w*\s*$", "", text)       # fences between files
    text = re.sub(r"(?m)^### ", "# ", text)            # multi-file headers echoed back
    ast.parse(text)  # raises if the model returned something that isn't Python
    return text + "\n"


def generate(prompt):
    """Python source, or None if the model's answer isn't valid Python (e.g. cut off).
    Responses are cached, so retrying the same prompt wouldn't help."""
    try:
        return clean(cards._chat(prompt, max_tokens=3000))
    except SyntaxError:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folders = [l.split("\t")[1] for l in git("ls-tree", "main").splitlines() if l.split()[1] == "tree"]
    jobs = {}
    for folder in folders:
        code = folder_code("main", folder)
        for style, desc in STYLES.items():
            jobs[f"{folder}__{style}"] = (REWRITE_PROMPT.format(style=desc, code=code), [folder])
    for name, task in NEGATIVE_TASKS.items():
        jobs[name] = (NEGATIVE_PROMPT.format(task=task), [])

    with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
        results = dict(zip(jobs, pool.map(lambda j: generate(jobs[j][0]), jobs)))
    skipped = [name for name, code in results.items() if code is None]
    results = {name: code for name, code in results.items() if code is not None}
    for name, code in results.items():
        (OUT / f"{name}.py").write_text(code)
    if skipped:
        print("skipped (invalid Python):", ", ".join(skipped))
    expected = {"_doc": "Generated by eval/gen_snippets.py. name -> {duplicate, related} folders on main (from the manifest); no duplicates = unrelated."}
    for name in results:
        if "__" in name:
            t = similarity_truth(name.split("__")[0])
            expected[name] = {"duplicate": sorted(set(t["duplicate"]) | set(t["partial"])), "related": []}
        else:
            expected[name] = NEGATIVE_LABELS.get(name, {"duplicate": [], "related": []})
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(results)} snippets to {OUT}")


if __name__ == "__main__":
    main()
