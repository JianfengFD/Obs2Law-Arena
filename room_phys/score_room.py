#!/usr/bin/env python3
"""
score_room.py — Scoring for Room Physics Challenge
=====================================================
Reads participant submissions and answer keys, computes scores.

Submission format (submit_room.txt):
    # One block per problem. Lines starting with # are comments.
    [a1b]
    g0 = 9.78
    alpha = -1.2

    [c3f]
    g0 = 9.5
    alpha = 3.0

Usage:
    python score_room.py --submission submit_room.txt --keys key_a1b.txt key_c3f.txt
    python score_room.py --submission submit_room.txt --key_dir ./keys/
"""

import os, sys, re, argparse, glob
import numpy as np


def parse_key_file(path):
    """Parse a key_XXX.txt file → dict with func_id, g0, alpha, seed_g."""
    info = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Function ID"):
                info["func_id"] = line.split(":")[-1].strip()
            elif line.startswith("seed_g"):
                info["seed_g"] = int(line.split(":")[-1].strip())
            elif line.startswith("Gravity"):
                # "Gravity     : g(z) = 9.773956 + -1.222431 / (z + 10)"
                m = re.search(r"g\(z\)\s*=\s*([-\d.]+)\s*\+\s*([-\d.]+)\s*/\s*\(z\s*\+\s*10\)", line)
                if m:
                    info["g0"] = float(m.group(1))
                    info["alpha"] = float(m.group(2))
    return info


def parse_submission(path):
    """Parse submit_room.txt → dict of {func_id: {g0, alpha}}."""
    results = {}
    current_id = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"\[(\w+)\]", line)
            if m:
                current_id = m.group(1)
                results[current_id] = {}
                continue
            if "=" in line and current_id:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                try:
                    results[current_id][key] = float(val.strip())
                except ValueError:
                    print(f"  Warning: cannot parse '{line}' in [{current_id}]")
    return results


def g_func(g0, alpha, z):
    """Compute g(z) = g0 + alpha / (z + 10)."""
    return g0 + alpha / (z + 10.0)


def score_one(submitted, truth, w_func=0.7, w_params=0.3):
    """
    Score a single problem using two complementary metrics:

    1. Functional score (w_func):
       mean |g_sub(z) - g_true(z)| over z in [0, 20], step 0.1
       Measures how well the predicted g(z) matches the truth.

    2. Parameter score (w_params):
       |g0_sub - g0_true| / |g0_true| + |alpha_sub - alpha_true| / max(|alpha_true|, 0.1)
       Measures how close the raw parameters are.

    total = w_func * func_score + w_params * param_score
    """
    g0_true = truth["g0"]
    alpha_true = truth["alpha"]

    g0_sub = submitted.get("g0")
    alpha_sub = submitted.get("alpha")

    result = {"g0_true": g0_true, "alpha_true": alpha_true}

    if g0_sub is not None and alpha_sub is not None:
        result["g0_sub"] = g0_sub
        result["alpha_sub"] = alpha_sub

        # Functional score
        z_arr = np.arange(0, 20.01, 0.1)
        g_true_arr = g_func(g0_true, alpha_true, z_arr)
        g_sub_arr = g_func(g0_sub, alpha_sub, z_arr)
        func_err = np.mean(np.abs(g_sub_arr - g_true_arr))
        result["func_err"] = func_err

        # Parameter score
        g0_rel = abs(g0_sub - g0_true) / abs(g0_true)
        alpha_rel = abs(alpha_sub - alpha_true) / max(abs(alpha_true), 0.1)
        param_err = g0_rel + alpha_rel
        result["g0_rel"] = g0_rel
        result["alpha_rel"] = alpha_rel
        result["param_err"] = param_err

        result["score"] = w_func * func_err + w_params * param_err
    else:
        # Missing submission — heavy penalty
        result["g0_sub"] = g0_sub
        result["alpha_sub"] = alpha_sub
        result["func_err"] = 10.0
        result["param_err"] = 10.0
        result["g0_rel"] = None
        result["alpha_rel"] = None
        result["score"] = 10.0

    return result


def main():
    ap = argparse.ArgumentParser(description="Score room physics submissions")
    ap.add_argument("--submission", type=str, required=True,
                    help="Path to submit_room.txt")
    ap.add_argument("--keys", type=str, nargs="*", default=None,
                    help="Path(s) to key_XXX.txt files")
    ap.add_argument("--key_dir", type=str, default=None,
                    help="Directory containing key_*.txt files (excluding key_telescope_*)")
    ap.add_argument("--w_func", type=float, default=0.7,
                    help="Weight for functional g(z) match score")
    ap.add_argument("--w_params", type=float, default=0.3,
                    help="Weight for parameter match score")
    args = ap.parse_args()

    # Collect key files
    key_files = []
    if args.keys:
        key_files.extend(args.keys)
    if args.key_dir:
        # Exclude telescope keys
        all_keys = sorted(glob.glob(os.path.join(args.key_dir, "key_*.txt")))
        key_files.extend([k for k in all_keys if "telescope" not in k])
    if not key_files:
        print("Error: no key files provided. Use --keys or --key_dir.")
        sys.exit(1)

    # Parse keys
    truths = {}
    for kf in key_files:
        info = parse_key_file(kf)
        fid = info.get("func_id")
        if fid and "g0" in info:
            truths[fid] = info
            print(f"  Key loaded: [{fid}] g0={info['g0']:.4f} alpha={info['alpha']:.4f}")

    # Parse submission
    submissions = parse_submission(args.submission)
    print(f"\nSubmission: {len(submissions)} problem(s) found")

    # Score
    scores = []
    print(f"\n{'='*90}")
    print(f"{'Problem':>8s}  {'g0_sub':>8s} {'g0_true':>8s} {'g0_rel':>7s}  "
          f"{'alpha_sub':>10s} {'alpha_true':>10s} {'alpha_rel':>9s}  "
          f"{'func_err':>8s}  {'SCORE':>8s}")
    print(f"{'-'*90}")

    for fid in sorted(set(list(truths.keys()) + list(submissions.keys()))):
        if fid not in truths:
            print(f"  [{fid}] — no answer key found, skipping")
            continue
        if fid not in submissions:
            print(f"  [{fid}] — not in submission, penalty applied")
            sub = {}
        else:
            sub = submissions[fid]

        r = score_one(sub, truths[fid], args.w_func, args.w_params)
        scores.append(r["score"])

        g0_s = f"{r['g0_sub']:.4f}" if r['g0_sub'] is not None else "MISSING"
        al_s = f"{r['alpha_sub']:.4f}" if r['alpha_sub'] is not None else "MISSING"
        g0r = f"{r['g0_rel']:.4f}" if r['g0_rel'] is not None else "  N/A"
        alr = f"{r['alpha_rel']:.4f}" if r['alpha_rel'] is not None else "    N/A"

        print(f"  [{fid:>4s}]  {g0_s:>8s} {r['g0_true']:>8.4f} {g0r:>7s}  "
              f"{al_s:>10s} {r['alpha_true']:>10.4f} {alr:>9s}  "
              f"{r['func_err']:>8.4f}  {r['score']:>8.4f}")

    print(f"{'='*90}")
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Average score: {avg:.4f}  (lower is better)")
        print(f"  Problems scored: {len(scores)}")
    print()


if __name__ == "__main__":
    main()
