"""Train one method with one seed, then evaluate it on all configured settings.

Example:
    python scripts/train.py --method protonet --seed 0
    python scripts/train.py --method relation --seed 0 --quick   # smoke test
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from fsl import compat  # noqa: E402,F401
from fsl.data import episodic_loader, load_omniglot  # noqa: E402
from fsl.engine import evaluate, train  # noqa: E402
from fsl.methods import METHOD_NAMES, build_method  # noqa: E402
from fsl.utils import count_parameters, get_device, load_config, save_json, seed_everything  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHOD_NAMES)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "raw"))
    parser.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr, ev = cfg["train"], cfg["eval"]
    if args.quick:
        tr["episodes"], ev["episodes"], ev["settings"] = 20, 5, [[5, 1]]

    device = get_device()
    seed_everything(args.seed)
    print(f"Method={args.method} | seed={args.seed} | device={device}")

    train_set, test_set = load_omniglot(cfg["data"]["root"], cfg["data"]["image_size"])
    method = build_method(args.method, cfg["methods"][args.method]).to(device)
    params = count_parameters(method)
    print(f"Parameters: {params['total']:,} total / {params['trainable']:,} trainable")

    optimizer = torch.optim.Adam(method.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    loader = episodic_loader(train_set, tr["n_way"], tr["k_shot"], tr["q_query"],
                             tr["episodes"], cfg["data"]["num_workers"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = train(method, loader, optimizer, device, tr["n_way"])
    train_seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    results = []
    for n_way, k_shot in ev["settings"]:
        res = evaluate(method, test_set, n_way, k_shot, ev["q_query"], ev["episodes"],
                       ev["episode_seed"], device, cfg["data"]["num_workers"])
        print(f"  {n_way}-way {k_shot}-shot: {100 * res['accuracy']:.2f}% +/- {100 * res['ci95']:.2f}%")
        results.append(res)

    out = {
        "method": args.method,
        "seed": args.seed,
        "params": params,
        "train_seconds": train_seconds,
        "peak_gpu_mb": peak_mb,
        "train_loss": history["loss"],
        "train_acc": history["acc"],
        "eval": results,
        "config": cfg,
    }
    save_json(out, Path(args.out_dir) / f"{args.method}_seed{args.seed}.json")
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(method.state_dict(), ckpt_dir / f"{args.method}_seed{args.seed}.pth")
    print("Saved results and checkpoint.")


if __name__ == "__main__":
    main()
