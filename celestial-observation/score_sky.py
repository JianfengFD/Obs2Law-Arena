#!/usr/bin/env python3
"""
score_sky.py — Scoring for Celestial Observation Challenge
============================================================
Reads participant submissions and answer keys, computes scores.

Submission format (submit_sky.txt):
    # One block per problem. Lines starting with # are comments.
    [T01]
    alpha = 0.012
    tilt = 23.5

    [T02]
    alpha = 0.005
    tilt = 24.1

Usage:
    python score_sky.py --submission submit_sky.txt --keys key_telescope_T01.txt key_telescope_T02.txt
    python score_sky.py --submission submit_sky.txt --key_dir ./keys/
"""

import os, sys, re, argparse, glob


def parse_key_file(path):
    """Parse a key_telescope_XXX.txt file → dict with func_id, alpha, tilt."""
    info = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Function ID"):
                info["func_id"] = line.split(":")[-1].strip()
            elif line.startswith("Alpha") and ":" in line:
                # "Alpha           : 0.01"
                info["alpha"] = float(line.split(":")[-1].strip())
            elif line.startswith("Tilt"):
                info["tilt"] = float(line.split(":")[-1].strip())
    return info


def parse_submission(path):
    """Parse submit_sky.txt → dict of {func_id: {alpha, tilt}}."""
    results = {}
    current_id = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Section header: [T01]
            m = re.match(r"\[(\w+)\]", line)
            if m:
                current_id = m.group(1)
                results[current_id] = {}
                continue
            # Key-value: alpha = 0.012
            if "=" in line and current_id:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                try:
                    results[current_id][key] = float(val.strip())
                except ValueError:
                    print(f"  Warning: cannot parse '{line}' in [{current_id}]")
    return results


def score_one(submitted, truth, w_alpha=0.8, w_tilt=0.2):
    """
    Score a single problem.

    Metrics:
      alpha_score = |alpha_sub - alpha_true| / max(|alpha_true|, 1e-6)
      tilt_score  = |tilt_sub - tilt_true|   (in degrees)
      total = w_alpha * alpha_score + w_tilt * tilt_score

    Returns dict with detailed breakdown.
    """
    alpha_true = truth["alpha"]
    tilt_true = truth["tilt"]
    alpha_sub = submitted.get("alpha")
    tilt_sub = submitted.get("tilt")

    result = {"alpha_true": alpha_true, "tilt_true": tilt_true}

    if alpha_sub is not None:
        alpha_err = abs(alpha_sub - alpha_true)
        alpha_rel = alpha_err / max(abs(alpha_true), 1e-6)
        result["alpha_sub"] = alpha_sub
        result["alpha_err"] = alpha_err
        result["alpha_rel"] = alpha_rel
    else:
        result["alpha_sub"] = None
        result["alpha_rel"] = 10.0  # penalty for missing

    if tilt_sub is not None:
        tilt_err = abs(tilt_sub - tilt_true)
        result["tilt_sub"] = tilt_sub
        result["tilt_err"] = tilt_err
    else:
        result["tilt_sub"] = None
        result["tilt_err"] = 30.0  # penalty for missing

    result["score"] = w_alpha * result["alpha_rel"] + w_tilt * result["tilt_err"]
    return result


def main():
    ap = argparse.ArgumentParser(description="Score sky observation submissions")
    ap.add_argument("--submission", type=str, required=True,
                    help="Path to submit_sky.txt")
    ap.add_argument("--keys", type=str, nargs="*", default=None,
                    help="Path(s) to key_telescope_XXX.txt files")
    ap.add_argument("--key_dir", type=str, default=None,
                    help="Directory containing key_telescope_*.txt files")
    ap.add_argument("--w_alpha", type=float, default=0.8)
    ap.add_argument("--w_tilt", type=float, default=0.2)
    args = ap.parse_args()

    # Collect key files
    key_files = []
    if args.keys:
        key_files.extend(args.keys)
    if args.key_dir:
        key_files.extend(sorted(glob.glob(
            os.path.join(args.key_dir, "key_telescope_*.txt"))))
    if not key_files:
        print("Error: no key files provided. Use --keys or --key_dir.")
        sys.exit(1)

    # Parse keys
    truths = {}
    for kf in key_files:
        info = parse_key_file(kf)
        fid = info.get("func_id")
        if fid:
            truths[fid] = info
            print(f"  Key loaded: [{fid}] alpha={info['alpha']} tilt={info['tilt']}")

    # Parse submission
    submissions = parse_submission(args.submission)
    print(f"\nSubmission: {len(submissions)} problem(s) found")

    # Score
    scores = []
    print(f"\n{'='*70}")
    print(f"{'Problem':>8s}  {'alpha_sub':>10s} {'alpha_true':>10s} {'alpha_rel':>10s}  "
          f"{'tilt_sub':>9s} {'tilt_true':>9s} {'tilt_err':>8s}  {'SCORE':>8s}")
    print(f"{'-'*70}")

    for fid in sorted(set(list(truths.keys()) + list(submissions.keys()))):
        if fid not in truths:
            print(f"  [{fid}] — no answer key found, skipping")
            continue
        if fid not in submissions:
            print(f"  [{fid}] — not in submission, penalty applied")
            sub = {}
        else:
            sub = submissions[fid]

        r = score_one(sub, truths[fid], args.w_alpha, args.w_tilt)
        scores.append(r["score"])

        a_sub = f"{r['alpha_sub']:.6f}" if r['alpha_sub'] is not None else "MISSING"
        t_sub = f"{r['tilt_sub']:.4f}" if r['tilt_sub'] is not None else "MISSING"
        print(f"  [{fid:>4s}]  {a_sub:>10s} {r['alpha_true']:>10.6f} {r['alpha_rel']:>10.4f}  "
              f"{t_sub:>9s} {r['tilt_true']:>9.4f} {r['tilt_err']:>8.4f}  {r['score']:>8.4f}")

    print(f"{'='*70}")
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Average score: {avg:.4f}  (lower is better)")
        print(f"  Problems scored: {len(scores)}")
    print()


if __name__ == "__main__":
    main()
