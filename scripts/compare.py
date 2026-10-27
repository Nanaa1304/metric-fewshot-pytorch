"""Aggregate results/raw/*.json into the final comparison report.

Outputs (in results/):
  summary.md / summary.csv      accuracy per method and setting + efficiency table
  pairwise.csv                  paired bootstrap differences between methods
  accuracy_comparison.png       grouped bar chart
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["siamese", "protonet", "matching", "relation"]
LABEL = {"siamese": "Siamese", "protonet": "ProtoNet", "matching": "Matching Net", "relation": "Relation Net"}


def load_runs(raw_dir: Path):
    runs = [json.load(open(p)) for p in sorted(raw_dir.glob("*_seed*.json"))]
    if not runs:
        raise SystemExit(f"No result files found in {raw_dir}. Run scripts/train.py first.")
    return runs


def accuracy_table(runs):
    rows = []
    for method in ORDER:
        mruns = [r for r in runs if r["method"] == method]
        if not mruns:
            continue
        settings = [(e["n_way"], e["k_shot"]) for e in mruns[0]["eval"]]
        for i, (n, k) in enumerate(settings):
            evs = [r["eval"][i] for r in mruns]
            seed_means = np.array([e["accuracy"] for e in evs]) * 100
            rows.append({
                "method": LABEL[method], "setting": f"{n}-way {k}-shot", "n_seeds": len(evs),
                "accuracy": seed_means.mean(),
                "seed_std": seed_means.std(ddof=1) if len(evs) > 1 else 0.0,
                "episode_ci95": np.mean([e["ci95"] for e in evs]) * 100,
                "precision": np.mean([e["precision"] for e in evs]) * 100,
                "recall": np.mean([e["recall"] for e in evs]) * 100,
                "f1": np.mean([e["f1"] for e in evs]) * 100,
            })
    return pd.DataFrame(rows)


def efficiency_table(runs):
    rows = []
    for method in ORDER:
        mruns = [r for r in runs if r["method"] == method]
        if not mruns:
            continue
        ms = [r["eval"][0]["ms_per_episode"] for r in mruns]
        peak = [r["peak_gpu_mb"] for r in mruns if r.get("peak_gpu_mb") is not None]
        rows.append({
            "method": LABEL[method],
            "params_M": mruns[0]["params"]["total"] / 1e6,
            "train_time_s": np.mean([r["train_seconds"] for r in mruns]),
            f"inference_ms_per_episode ({mruns[0]['eval'][0]['n_way']}-way {mruns[0]['eval'][0]['k_shot']}-shot)": np.mean(ms),
            "peak_gpu_MB_train": np.mean(peak) if peak else float("nan"),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(runs, n_boot=10000, rng_seed=0):
    """Same test episodes for every method (shared episode_seed) => per-episode paired
    differences. Pool episodes over seeds, bootstrap the mean difference."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    sample = next(r for r in runs)
    settings = [(e["n_way"], e["k_shot"]) for e in sample["eval"]]
    for i, (n, k) in enumerate(settings):
        acc = {}
        for method in ORDER:
            per_seed = {r["seed"]: np.array(r["eval"][i]["episode_accuracies"])
                        for r in runs if r["method"] == method}
            if per_seed:
                acc[method] = per_seed
        for a, b in itertools.combinations([m for m in ORDER if m in acc], 2):
            common = sorted(set(acc[a]) & set(acc[b]))
            if not common:
                continue
            diff = np.concatenate([acc[a][s] - acc[b][s] for s in common]) * 100
            boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "setting": f"{n}-way {k}-shot", "A": LABEL[a], "B": LABEL[b],
                "mean_diff_pp (A-B)": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                "significant": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


def plot(acc_df, path):
    settings = list(dict.fromkeys(acc_df["setting"]))
    methods = list(dict.fromkeys(acc_df["method"]))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, m in enumerate(methods):
        sub = acc_df[acc_df["method"] == m].set_index("setting").loc[settings]
        ax.bar(np.arange(len(settings)) + j * width, sub["accuracy"], width,
               yerr=sub["seed_std"], capsize=3, label=m)
    ax.set_xticks(np.arange(len(settings)) + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(settings)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, acc_df["accuracy"].min() - 10), 100)
    ax.set_title("Metric-based few-shot methods on Omniglot (ResNet18, unseen classes)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "results" / "raw"))
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(Path(args.raw_dir))
    acc_df, eff_df, pair_df = accuracy_table(runs), efficiency_table(runs), paired_bootstrap(runs)

    acc_df.to_csv(out / "summary.csv", index=False)
    pair_df.to_csv(out / "pairwise.csv", index=False)
    plot(acc_df, out / "accuracy_comparison.png")

    # Compact markdown: accuracy "mean ± seed-std"
    pivot = acc_df.assign(
        cell=lambda d: d.apply(lambda r: f"{r['accuracy']:.2f} ± {r['seed_std']:.2f}", axis=1)
    ).pivot(index="method", columns="setting", values="cell").reindex([LABEL[m] for m in ORDER if LABEL[m] in set(acc_df['method'])])

    with open(out / "summary.md", "w") as f:
        f.write("## Accuracy (%) — mean ± std over training seeds\n\n")
        f.write(pivot.to_markdown() + "\n\n")
        f.write("## Macro precision / recall / F1 (%) — mean over seeds\n\n")
        f.write(acc_df[["method", "setting", "precision", "recall", "f1"]].round(2).to_markdown(index=False) + "\n\n")
        f.write("## Efficiency\n\n" + eff_df.round(2).to_markdown(index=False) + "\n\n")
        f.write("## Paired bootstrap differences (percentage points)\n\n")
        f.write(pair_df.round(2).to_markdown(index=False) + "\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
