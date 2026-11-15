#!/usr/bin/env python
"""Aggregate evaluate_full_neural_vs_analytical.py logs over several training seeds.

Usage (from the repository root):
    .venv/bin/python reproducibility/aggregate_seeds.py runs/seed_* \
        [--committed runs/committed/eval.log] [--diverged-above 0.5] [--json runs/aggregate.json]

Each run directory must contain eval.log and nn-results/<net>/checkpoints/*.pt
(as produced by reproducibility/multiseed.sh). Directories from several machines
can be passed together to pool them; they must have distinct names.

A network whose global MPJPE exceeds --diverged-above metres in a run is treated as a
diverged training run: it is listed, and excluded from that network's mean/SD and from
the paired TCN-vs-Transformer test. Pass --diverged-above inf to keep every run.
"""
import sys, re, json, glob, argparse
from pathlib import Path
import numpy as np

METHODS = ["Static", "Oracle", "TCN Lite", "TTL Lite", "Transformer", "Analytical"]
BLOCKS = ["rel_mpjpe", "mpjpe", "mae", "rmse"]  # order of GENERAL lines in the log
CKPTS = {
    "PivotNet": "nn-results/pivot_classifier/checkpoints/best_pivot_classifier.pt",
    "Static": "nn-results/static_pivot/checkpoints/best_static_model.pt",
    "TCN Lite": "nn-results/tcn_lite/checkpoints/best_tcn_lite.pt",
    "TTL Lite": "nn-results/ttl_lite/checkpoints/best_ttl_lite.pt",
    "Transformer": "nn-results/transformer_multitask/checkpoints/best_multitask_model_trans.pt",
}


def parse_log(path):
    txt = Path(path).read_text(errors="replace")
    gen = re.findall(r"^\s+GENERAL\s+(.*)$", txt, re.M)
    if len(gen) < 4:
        raise ValueError(f"{path}: expected 4 GENERAL lines, found {len(gen)}")
    out = {}
    for name, line in zip(BLOCKS, gen[:4]):
        vals = [float(v) for v in re.findall(r"(-?\d+\.\d+)\s*m", line)]
        out[name] = dict(zip(METHODS, vals))
    m = re.search(r"PivotNet classification accuracy:\s*([\d.]+)%", txt)
    out["pivot_acc"] = float(m.group(1)) if m else None
    return out


def ckpt_epochs(d):
    import torch
    ep = {}
    for k, rel in CKPTS.items():
        p = Path(d) / rel
        if p.exists():
            c = torch.load(p, map_location="cpu", weights_only=False)
            ep[k] = c.get("epoch") if isinstance(c, dict) else None
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--committed")
    ap.add_argument("--json")
    ap.add_argument("--diverged-above", type=float, default=0.5,
                    help="global MPJPE (m) above which a network's run counts as diverged (default 0.5)")
    a = ap.parse_args()

    runs = {}
    for d in a.dirs:
        log = Path(d) / "eval.log"
        if not log.exists():
            print(f"skip {d}: no eval.log"); continue
        try:
            r = parse_log(log)
        except ValueError as e:
            print(f"skip {d}: {e}"); continue
        r["epochs"] = ckpt_epochs(d)
        name = Path(d).name
        if name in runs:
            name = f"{Path(d).parent.name}/{name}"
        runs[name] = r
    seeds = sorted(runs, key=lambda s: (int(re.sub(r"\D", "", Path(s).name) or 0), s))
    n = len(seeds)
    print(f"Runs aggregated (n={n}): {', '.join(seeds)}\n")

    thr = a.diverged_above
    diverged = {m: [s for s in seeds if runs[s]["mpjpe"][m] > thr] for m in METHODS if m not in ("Static", "Oracle", "Analytical")}
    if any(diverged.values()):
        print(f"Diverged runs (global MPJPE > {thr} m), excluded from mean/SD:")
        for m, ss in diverged.items():
            for s in ss:
                print(f"  {m} {s}: {runs[s]['mpjpe'][m]:.3f} m")
        print()
    def kept(m):
        return [s for s in seeds if s not in diverged.get(m, [])]

    committed = parse_log(a.committed) if a.committed else None
    summary = {}
    for blk in BLOCKS:
        print(f"== {blk} (m) ==")
        hdr = f"{'method':<12} " + " ".join(f"{s[-9:]:>9} " for s in seeds) + f" {'n':>3} {'mean':>8} {'SD':>7} {'min':>7} {'max':>7}"
        if committed: hdr += f" {'committed':>9}"
        print(hdr)
        summary[blk] = {}
        for m in METHODS:
            allv = np.array([runs[s][blk][m] for s in seeds])
            v = np.array([runs[s][blk][m] for s in kept(m)])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            summary[blk][m] = {"values": allv.tolist(), "n": len(v), "excluded": diverged.get(m, []),
                               "mean": v.mean(), "sd": sd, "min": v.min(), "max": v.max()}
            line = f"{m:<12} " + " ".join(f"{x:9.4f}" + ("*" if s in diverged.get(m, []) else " ") for x, s in zip(allv, seeds)) \
                + f" {len(v):3d} {v.mean():8.4f} {sd:7.4f} {v.min():7.4f} {v.max():7.4f}"
            if committed: line += f" {committed[blk][m]:9.4f}"
            print(line)
        print()
    if any(diverged.values()):
        print("(* = diverged run, not included in n/mean/SD/min/max)\n")

    # relative improvement vs analytical, per seed then mean ± SD
    print("== Δ global MPJPE vs analytical (%), computed per seed ==")
    summary["delta_mpjpe_pct"] = {}
    for m in METHODS[:-1]:
        d = np.array([(runs[s]["mpjpe"]["Analytical"] - runs[s]["mpjpe"][m]) / runs[s]["mpjpe"]["Analytical"] * 100 for s in kept(m)])
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        summary["delta_mpjpe_pct"][m] = {"values": d.tolist(), "mean": d.mean(), "sd": sd}
        print(f"{m:<12} {d.mean():+7.2f} ± {sd:5.2f}   [" + ", ".join(f"{x:+.1f}" for x in d) + "]")

    print("\n== Winner (lowest global MPJPE among NN) per seed ==")
    nn = ["TCN Lite", "TTL Lite", "Transformer"]
    for s in seeds:
        best = min(nn, key=lambda m: runs[s]["mpjpe"][m])
        print(f"  {s}: {best}  (" + ", ".join(f"{m} {runs[s]['mpjpe'][m]:.4f}" for m in nn) + ")")

    # paired comparison TCN vs Transformer
    both = [s for s in kept("TCN Lite") if s in kept("Transformer")]
    t = np.array([runs[s]["mpjpe"]["TCN Lite"] for s in both])
    tr = np.array([runs[s]["mpjpe"]["Transformer"] for s in both])
    if len(both) > 1:
        diff = t - tr
        print(f"\nTCN − Transformer global MPJPE: mean {diff.mean():+.4f} m, SD {diff.std(ddof=1):.4f} m (n={len(both)} paired runs)")
        try:
            from scipy.stats import ttest_rel, wilcoxon
            print(f"  paired t-test p={ttest_rel(t, tr).pvalue:.4f}" + (f", Wilcoxon p={wilcoxon(t, tr).pvalue:.4f}" if len(both) >= 5 else ""))
        except Exception as e:
            print("  (stats unavailable:", e, ")")

    print("\n== Best epochs per seed ==")
    for s in seeds:
        print(f"  {s}: " + ", ".join(f"{k} {v}" for k, v in runs[s]["epochs"].items()) + f" | PivotNet acc {runs[s]['pivot_acc']}%")

    if a.json:
        Path(a.json).write_text(json.dumps({"seeds": seeds, "runs": runs, "summary": summary}, indent=2, default=float))
        print(f"\nSaved {a.json}")


if __name__ == "__main__":
    main()
