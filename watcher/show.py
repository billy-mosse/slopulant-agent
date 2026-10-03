"""Prints the latest candidates per PR (what the duplicate detector will receive).

    python -m watcher.show          # candidates only
    python -m watcher.show --all    # every scored pair
"""
import sys

from . import db


def main():
    conn = db.connect()
    where = "" if "--all" in sys.argv else "WHERE candidate = 1"
    rows = conn.execute(f"SELECT * FROM latest_scores {where} ORDER BY pr_number, pr_folder, rank").fetchall()
    current = None
    for r in rows:
        if (r["pr_number"], r["pr_folder"]) != current:
            current = (r["pr_number"], r["pr_folder"])
            print(f"\nPR #{r['pr_number']} @ {r['head_sha'][:7]}  {r['pr_folder']}/")
        why = (f"{r['dataflow']}; " if r["dataflow"] else "") + (
            f"desc {r['card_score']:.2f}, kw {r['kw_score']:.2f} [{r['kw_match'] or '-'}], "
            f"code {r['code_score']:.2f} ({r['code_match']})"
        )
        print(f"  {r['rank']:>2}. {r['repo_id']:<22} {r['score']:.2f}  {why}")


if __name__ == "__main__":
    main()
