#!/usr/bin/env python3
"""Summarize bench results.csv by provider+model.
Excludes warmup runs. Reports mean decode/prefill and median TTFT."""
import argparse, csv, statistics as st
from collections import defaultdict


def num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    args = ap.parse_args()

    rows = defaultdict(list)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("warmup", "").strip().lower() == "true":
                continue  # skip warmup runs
            provider = r.get("provider", "") or "?"
            key = f"{provider}/{r['model']}"
            rows[key].append(r)

    if not rows:
        print("No non-warmup rows found.")
        return

    print(f"\n{'Provider / Model':<28} {'runs':>4} {'decode tok/s':>15} "
          f"{'prefill tok/s':>15} {'TTFT ms (p50)':>15}")
    print("-" * 80)
    for key, rs in sorted(rows.items()):
        dec = [num(r["decode_tps"]) for r in rs if num(r["decode_tps"])]
        pre = [num(r["prefill_tps"]) for r in rs if num(r["prefill_tps"])]
        ttft = [num(r["ttft_ms"]) for r in rs if num(r["ttft_ms"])]
        if dec:
            dec_s = f"{st.mean(dec):.1f}" + (
                f" \u00b1{st.pstdev(dec):.1f}" if len(dec) > 1 else "")
        else:
            dec_s = "n/a"
        pre_s = f"{st.mean(pre):.0f}" if pre else "n/a"
        ttft_s = f"{st.median(ttft):.0f}" if ttft else "n/a"
        print(f"{key:<28} {len(rs):>4} {dec_s:>15} {pre_s:>15} {ttft_s:>15}")
    print()


if __name__ == "__main__":
    main()