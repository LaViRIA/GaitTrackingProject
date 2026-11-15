"""
Execution-time benchmark for the manuscript's computational-cost claims.

Measures, on the held-out test trials (data_split.json) and with the committed
checkpoints, three things:

  A. STREAMING latency of one inference step, i.e. what a real-time system pays
     every time a new 25-sample window is ready:
        normalise window -> append anthropometry -> host->device copy ->
        forward pass (fp16 autocast on CUDA, as in inference) -> device->host copy.
     Plus the per-sample cost of a *causal* 6th-order Butterworth input filter
     (sosfilt with state), the online replacement of the offline filtfilt.
     Batch size 1, measured call by call with time.perf_counter() after a
     CUDA synchronize.  Reported against the 10 ms budget of a 100 Hz stream.

  B. OFFLINE throughput of the repository's own inference routine
     (predict_full_sequence_multitask, stride = window/2) over each full test
     trial, i.e. exactly how the reported accuracy numbers are produced.

  C. ANALYTICAL pipeline cost per trial, split into
        read_mcd_file  (CSV parsing + unit conversion + zero-phase filter)
        mcd_reconstruction  (orientation fusion, alignment, kinematic chain,
                             support-foot heuristic, displacement chaining)
     with the hyper-parameters used by evaluate_full_neural_vs_analytical.py.
     Single process.  Ground-truth reading/alignment is NOT timed.

Every timing reports n, mean, std, median, p95, p99 and max (milliseconds).
Results are printed and written to a JSON file together with the hardware and
software environment, so the numbers can be cited verbatim.

Usage (from the repository root):
    python benchmark_timing.py --dataset /path/to/DataSet
    python benchmark_timing.py --dataset ... --devices cpu --cpu-threads 1
    python benchmark_timing.py --dataset ... --out nn-results/timing/timing_rtx5080.json

For stable numbers: close other GPU/CPU workloads, keep the machine on AC
power / performance governor, and run the script twice to check consistency.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_full_neural_vs_analytical as ev  # noqa: E402  (checkpoint paths)
import models.nn_tcn_lite as tcn_mod  # noqa: E402
import models.nn_ttl_lite as ttl_mod  # noqa: E402
import models.nn_transformer_multitask as trans_mod  # noqa: E402
from models.nn_static_pivot import (  # noqa: E402
    GaitStaticDataset,
    discover_captures,
    prepare_static_capture,
)
from process_all import (  # noqa: E402
    adjust_aruco_data,
    mcd_reconstruction,
    measures_calculation,
    read_aruco_file,
    read_mcd_file,
)

# Hyper-parameters of the analytical pipeline as used in the evaluation script.
ANALYTICAL_PARAMS = dict(
    cutoff_freq=9,
    filter_order=6,
    pca_percentile=20,
    madgwick_gain=0.02,
    slerp_clip_min=0.5,
    slerp_clip_max=0.5,
    calibration_samples=100,
    fusion_filter="madgwick",
    pivot_height_threshold=0.005,
    swing_velocity_threshold=0.3,
)
NN_CUTOFF_HZ = tcn_mod.CUTOFF_FREQ  # 7 Hz
NN_FILTER_ORDER = tcn_mod.FILTER_ORDER  # 6
FS_HZ = 100.0
BONE_NORM = GaitStaticDataset.BONE_NORM

MODELS = {
    "tcn": ("TCN Lite", tcn_mod, ev.TCN_CKPT_PATH, ev.TCN_STATS_PATH),
    "ttl": ("TTL Lite", ttl_mod, ev.TTL_CKPT_PATH, ev.TTL_STATS_PATH),
    "trans": ("Transformer", trans_mod, ev.TRANS_CKPT_PATH, None),
}


# ───────────────────────────── helpers ─────────────────────────────


def stats_ms(samples_s: list[float]) -> dict:
    a = np.asarray(samples_s, dtype=np.float64) * 1e3
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def fmt(s: dict, unit="ms") -> str:
    return (
        f"median {s['median']:.3f} {unit} | mean {s['mean']:.3f} ± {s['std']:.3f} | "
        f"p95 {s['p95']:.3f} | p99 {s['p99']:.3f} | max {s['max']:.3f} (n={s['n']})"
    )


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment(args) -> dict:
    env = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch_default_threads": torch.get_num_threads(),
        "args": vars(args),
    }
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        env["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        env["git_commit"] = None
    return env


def load_model(key: str, device: torch.device):
    """Rebuild a sequence model from its checkpoint exactly as the evaluator does."""
    name, mod, ckpt_path, stats_path = MODELS[key]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config") or {}
    in_dim = state["input_proj.0.weight"].shape[1]

    def tcn_arch():
        n = sum(1 for k in state if k.startswith("tcn_blocks.") and k.endswith(".conv.weight"))
        ch = [state[f"tcn_blocks.{i}.conv.weight"].shape[0] for i in range(n)]
        ks = [state[f"tcn_blocks.{i}.conv.weight"].shape[2] for i in range(n)]
        return ch, ks, cfg.get("tcn_dilations", [2**i for i in range(n)])

    def lstm_attn():
        lstm_h = state["lstm.weight_hh_l0"].shape[1]
        lstm_l = sum(1 for k in state if k.startswith("lstm.weight_ih_l") and "reverse" not in k)
        attn_l = sum(
            1 for k in state if k.startswith("attention.layers.") and k.endswith(".norm1.weight")
        )
        return lstm_h, lstm_l, attn_l

    if key == "tcn":
        ch, ks, dil = tcn_arch()
        model = mod.GaitMultiTaskNetTCN(
            input_dim=in_dim,
            tcn_channels=ch,
            tcn_kernel_sizes=ks,
            tcn_dilations=dil,
            head_hidden_dim=state["pose_head.1.weight"].shape[0],
        )
    elif key == "ttl":
        ch, ks, dil = tcn_arch()
        lstm_h, lstm_l, attn_l = lstm_attn()
        model = mod.GaitMultiTaskNetTTLLite(
            input_dim=in_dim,
            tcn_channels=ch,
            tcn_kernel_sizes=ks,
            tcn_dilations=dil,
            lstm_hidden=lstm_h,
            lstm_layers=lstm_l,
            n_heads=cfg.get("n_heads", 2),
            n_attn_layers=attn_l,
            head_hidden_dim=state["pose_head.1.weight"].shape[0],
        )
    else:
        cnn = []
        for i in range(10):
            k = f"cnn_blocks.{i}.conv.weight"
            if k not in state:
                break
            cnn.append(state[k].shape[0])
        lstm_h, lstm_l, attn_l = lstm_attn()
        model = mod.GaitMultiTaskNet(
            input_dim=in_dim,
            cnn_channels=cnn,
            lstm_hidden=lstm_h,
            lstm_layers=lstm_l,
            n_heads=cfg.get("n_heads", 12),
            n_attn_layers=attn_l,
        )
    model.load_state_dict(state)
    model.to(device).eval()

    ns = ckpt.get("norm_stats")
    if ns is None and stats_path is not None and Path(stats_path).exists():
        ns = json.loads(Path(stats_path).read_text())
    if ns is None:  # evaluator falls back to the TCN statistics
        ns = json.loads(Path(ev.TCN_STATS_PATH).read_text())
    ns = {
        k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)).astype(np.float32)
        for k, v in ns.items()
        if k in ("mean", "std")
    }
    window = int(cfg.get("window_size", 25))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "name": name,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_short(Path(ckpt_path)),
        "trainable_params": n_params,
        "input_dim": in_dim,
        "window": window,
    }
    return model, ns, window, info


def anthropometry(cap: dict, in_dim: int) -> np.ndarray:
    d = 1.0 if cap["direction"] == "R" else 0.0
    if in_dim == 28:
        v = [cap["femur_length"], cap["tibia_length"], cap["hip_width"]]
    else:
        v = [cap["femur_length"], cap["tibia_length"]]
    return np.array([x / BONE_NORM for x in v] + [d], dtype=np.float32)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ───────────────────────────── A. streaming ─────────────────────────────


def bench_stream(model, ns, window, caps, device, iters, warmup) -> dict:
    """One inference step per call, batch 1, on real test windows."""
    in_dim = model.input_proj[0].weight.shape[1]
    windows = []
    for cap in caps:
        x = cap["sensor_data"].astype(np.float32)
        a = anthropometry(cap, in_dim)
        for s in range(0, len(x) - window + 1, window):
            windows.append((x[s : s + window], a))
    use_amp = device.type == "cuda"

    def step(raw, a):
        xn = (raw - ns["mean"]) / ns["std"]
        inp = np.concatenate([xn, np.broadcast_to(a, (window, a.size))], axis=1)
        t = torch.from_numpy(np.ascontiguousarray(inp, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=use_amp):
            pose, disp = model(t)
        return pose.float().cpu().numpy(), disp.float().cpu().numpy()

    times = []
    for i in range(warmup + iters):
        raw, a = windows[i % len(windows)]
        sync(device)
        t0 = time.perf_counter()
        step(raw, a)
        sync(device)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return stats_ms(times)


def bench_causal_filter(caps, iters, warmup) -> dict:
    """Per-sample causal Butterworth on the 24 IMU channels (online equivalent of filtfilt)."""
    sos = scipy.signal.butter(NN_FILTER_ORDER, NN_CUTOFF_HZ, fs=FS_HZ, output="sos")
    x = np.concatenate([c["sensor_data"] for c in caps], axis=0).astype(np.float64)
    zi = np.zeros((sos.shape[0], 2, x.shape[1]))
    times = []
    for i in range(warmup + iters):
        sample = x[i % len(x)][None, :]
        t0 = time.perf_counter()
        _, zi = scipy.signal.sosfilt(sos, sample, axis=0, zi=zi)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return stats_ms(times)


# ───────────────────────────── B. offline ─────────────────────────────


def bench_offline(key, model, ns, window, caps, device, repeats) -> dict:
    """Repository inference routine over complete trials (as in the evaluation)."""
    mod = MODELS[key][1]
    old_dev = mod.DEVICE
    mod.DEVICE = device  # the routine moves inputs to its module-level DEVICE
    in_dim = model.input_proj[0].weight.shape[1]
    try:
        with torch.inference_mode():
            c0 = caps[0]
            mod.predict_full_sequence_multitask(
                model, c0["sensor_data"], anthropometry(c0, in_dim), ns, window_size=window
            )  # warm-up
            per_trial, frames = [], []
            for cap in caps:
                reps = []
                for _ in range(repeats):
                    sync(device)
                    t0 = time.perf_counter()
                    mod.predict_full_sequence_multitask(
                        model,
                        cap["sensor_data"],
                        anthropometry(cap, in_dim),
                        ns,
                        window_size=window,
                    )
                    sync(device)
                    reps.append(time.perf_counter() - t0)
                per_trial.append(float(np.median(reps)))
                frames.append(len(cap["sensor_data"]))
    finally:
        mod.DEVICE = old_dev
    total_s, total_f = sum(per_trial), sum(frames)
    return {
        "trials": len(per_trial),
        "frames": int(total_f),
        "total_ms": total_s * 1e3,
        "ms_per_frame": total_s * 1e3 / total_f,
        "per_trial_ms": stats_ms(per_trial),
        "stride": window // 2,
    }


# ───────────────────────────── C. analytical ─────────────────────────────


def bench_analytical(files, repeats) -> dict:
    p = ANALYTICAL_PARAMS
    rec_params = {k: v for k, v in p.items() if k not in ("cutoff_freq", "filter_order")}
    t_read, t_rec, frames = [], [], []
    for mcd_path, aruco_path in files:
        reads = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mcd = read_mcd_file(mcd_path, cutoff_freq=p["cutoff_freq"], filter_order=p["filter_order"])
            reads.append(time.perf_counter() - t0)
        # untimed set-up, identical to process_files(): segment lengths, hip width
        direction = mcd["measures"]["direction"]
        aruco = adjust_aruco_data(read_aruco_file(aruco_path), direction)
        ap = {
            "hip_r": aruco["hip_position"],
            "hip_l": aruco["hip_position"],
            "knee_r": aruco["R_knee_position"],
            "knee_l": aruco["L_knee_position"],
            "ankle_r": aruco["R_ankle_position"],
            "ankle_l": aruco["L_ankle_position"],
        }
        for k in ap:
            ap[k]["data"][:, 1] = 0
        fl, tl = measures_calculation(ap, direction)
        mcd["measures"]["femur_length"], mcd["measures"]["tibia_length"] = fl, tl
        if mcd["measures"].get("hip_width", 0) <= 0.05:
            mcd["measures"]["hip_width"] = 0.25
        recs = []
        for _ in range(repeats):
            data = copy.deepcopy(mcd)
            t0 = time.perf_counter()
            mcd_reconstruction(data, **rec_params)
            recs.append(time.perf_counter() - t0)
        t_read.append(float(np.median(reads)))
        t_rec.append(float(np.median(recs)))
        frames.append(len(mcd["time"]))
    n = sum(frames)
    return {
        "trials": len(frames),
        "frames": int(n),
        "read_filter_ms_per_frame": sum(t_read) * 1e3 / n,
        "reconstruction_ms_per_frame": sum(t_rec) * 1e3 / n,
        "total_ms_per_frame": (sum(t_read) + sum(t_rec)) * 1e3 / n,
        "reconstruction_per_trial_ms": stats_ms(t_rec),
        "read_filter_per_trial_ms": stats_ms(t_read),
        "params": ANALYTICAL_PARAMS,
    }


# ───────────────────────────── main ─────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="DataSet", help="DataSet root (subject folders; default: DataSet/)")
    ap.add_argument("--data-split", default="data_split.json")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument(
        "--devices",
        nargs="+",
        default=(["cuda"] if torch.cuda.is_available() else []) + ["cpu"],
        choices=["cuda", "cpu"],
    )
    ap.add_argument(
        "--cpu-threads",
        nargs="+",
        type=int,
        default=[torch.get_num_threads(), 1],
        help="torch intra-op thread counts to test on CPU (default: all, 1)",
    )
    ap.add_argument("--iters", type=int, default=2000, help="timed streaming steps")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="repeats per trial (offline/analytical)")
    ap.add_argument("--skip-offline", action="store_true")
    ap.add_argument("--skip-analytical", action="store_true")
    ap.add_argument("--out", default="nn-results/timing/timing_results.json")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    env = environment(args)
    print("Environment:")
    for k in ("cpu", "logical_cpus", "gpu", "torch", "cuda_runtime", "python", "git_commit"):
        print(f"  {k}: {env.get(k)}")

    split = json.loads(Path(args.data_split).read_text())
    test = {str(s) for s in split["test_subjects"]}
    files, caps = [], []
    for mcd_path, aruco_path, subj in discover_captures(Path(args.dataset)):
        if subj in test:
            cap = prepare_static_capture(mcd_path, aruco_path)
            if cap is not None:
                files.append((mcd_path, aruco_path))
                caps.append(cap)
    print(f"Test trials: {len(caps)} ({sum(len(c['sensor_data']) for c in caps)} frames)\n")

    results = {"environment": env, "models": {}, "stream": [], "offline": [], "budget_ms_at_100Hz": 10.0}

    print("── A0. Causal input filter, one 24-channel sample ──")
    results["causal_filter_per_sample"] = bench_causal_filter(caps, args.iters, args.warmup)
    print("  " + fmt(results["causal_filter_per_sample"]) + "\n")

    configs = []
    for dev in args.devices:
        if dev == "cuda":
            if not torch.cuda.is_available():
                print("CUDA not available, skipping")
                continue
            configs.append(("cuda", None))
        else:
            configs += [("cpu", t) for t in args.cpu_threads]

    default_threads = torch.get_num_threads()
    for dev_name, threads in configs:
        device = torch.device(dev_name)
        torch.set_num_threads(threads or default_threads)
        label = f"{dev_name}" + (f" ({threads} thread{'s' if threads != 1 else ''})" if threads else "")
        for key in args.models:
            model, ns, window, info = load_model(key, device)
            results["models"][key] = info
            s = bench_stream(model, ns, window, caps, device, args.iters, args.warmup)
            results["stream"].append({"model": key, "device": label, **s})
            print(f"── A. Streaming step, {info['name']}, {label} ──\n  {fmt(s)}")
            if not args.skip_offline:
                o = bench_offline(key, model, ns, window, caps, device, args.repeats)
                results["offline"].append({"model": key, "device": label, **o})
                print(
                    f"   B. Offline full trials: {o['ms_per_frame']*1e3:.2f} µs/frame "
                    f"({o['frames']} frames, stride {o['stride']}, "
                    f"median {o['per_trial_ms']['median']:.1f} ms/trial)"
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print()
    torch.set_num_threads(default_threads)

    if not args.skip_analytical:
        print("── C. Analytical pipeline (single process) ──")
        a = bench_analytical(files, args.repeats)
        results["analytical"] = a
        print(
            f"  read+filter {a['read_filter_ms_per_frame']:.3f} ms/frame | "
            f"reconstruction {a['reconstruction_ms_per_frame']:.3f} ms/frame | "
            f"total {a['total_ms_per_frame']:.3f} ms/frame ({a['frames']} frames, {a['trials']} trials)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
