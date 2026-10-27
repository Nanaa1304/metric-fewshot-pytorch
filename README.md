# Metric-Based Few-Shot Learning: A Controlled Comparison on Omniglot

A small research project comparing the four classical **metric-based** few-shot learning methods under
**one shared protocol** (same dataset, same ResNet18 backbone, same training budget, same test episodes):

| Method | Paper | Similarity / comparison |
|---|---|---|
| **Siamese Network** | Koch et al., 2015 | Euclidean distance, contrastive loss on pairs |
| **Prototypical Network** | Snell et al., 2017 | Squared Euclidean distance to class prototypes |
| **Matching Network** | Vinyals et al., 2016 | Cosine attention + Full Context Embeddings (BiLSTM) |
| **Relation Network** | Sung et al., 2018 | *Learned* comparison with a CNN relation module |

The central question: **does it matter whether the metric is fixed (Euclidean, cosine) or learned, and
how much does each design add in accuracy, parameters and compute?**

> Status: the code and protocol are complete. Run `scripts/run_all.sh` to produce the final results table
> (see [Results](#results)). Exploratory single-run numbers from the original notebooks are in
> [`results/preliminary_notebook_results.md`](results/preliminary_notebook_results.md).

## Few-shot setting in one paragraph

Training and testing are **episodic**. An episode is an *N-way K-shot* task: N classes are drawn at random,
K labelled images per class form the **support set**, and Q further images per class form the **query set**
to be classified. Models are meta-trained on the 964 Omniglot "background" classes and evaluated on the 659
**unseen** "evaluation" classes, so accuracy measures generalisation to new classes.

## Experimental protocol

| Factor | Value |
|---|---|
| Dataset | Omniglot (torchvision official split), 84×84 |
| Backbone | ResNet18, random init |
| Training | 1000 episodes, 5-way 1-shot, 5 queries/class, Adam (lr 1e-4, wd 1e-5) |
| Evaluation | 5-way/10-way × 1-shot/5-shot, 600 episodes, 15 queries/class, **identical episodes for all methods** |
| Repetitions | 3 seeds per method |
| Metrics | accuracy (± seed std, episode 95% CI), macro precision/recall/F1, #parameters, training time, inference time, peak GPU memory, paired bootstrap tests |

Full details, the table of intentional differences between methods, and the list of changes relative to the
original notebooks are in [`docs/methodology.md`](docs/methodology.md).

## Repository structure

```
.
├── configs/default.yaml        # the single source of truth for the protocol
├── src/fsl/
│   ├── data.py                 # Omniglot + episodic loaders (easyfsl TaskSampler)
│   ├── backbones.py            # ResNet18 (vector embedding / feature map)
│   ├── engine.py               # shared train & evaluate loops
│   ├── compat.py               # easyfsl 1.5.0 <-> recent PyTorch patch
│   └── methods/
│       ├── siamese.py          # from scratch
│       ├── protonet.py         # from scratch
│       ├── matching.py         # easyfsl.MatchingNetworks
│       └── relation.py         # easyfsl.RelationNetworks
├── scripts/
│   ├── train.py                # train + evaluate one method / one seed
│   ├── run_all.sh              # 4 methods x all seeds, then compare
│   └── compare.py              # aggregate -> tables, plot, paired bootstrap
├── notebooks/original/         # the 4 exploratory notebooks this project grew from
├── results/                    # raw JSON (git-ignored), summary.md/csv, figures
└── docs/methodology.md
```

## Quick start

```bash
git clone https://github.com/<your-username>/metric-fewshot-benchmark.git
cd metric-fewshot-benchmark
pip install -r requirements.txt

# 1. Smoke test (about a minute): 20 training episodes, 5 test episodes
python scripts/train.py --method protonet --seed 0 --quick

# 2. One method, one seed
python scripts/train.py --method relation --seed 0

# 3. Full benchmark (4 methods x 3 seeds) + report
bash scripts/run_all.sh
```

A GPU is strongly recommended (the original experiments ran on a Colab Tesla T4). Omniglot is downloaded
automatically by torchvision on first use.

## Results

Run `bash scripts/run_all.sh`; it writes `results/summary.md`, `results/summary.csv`,
`results/pairwise.csv` and `results/accuracy_comparison.png`. Paste the final table here:

Method	5-way 1-shot	5-way 5-shot	Gain	Time reported
Siamese Network	72.19	81.91	+9.72	64 s
Prototypical Network	87.96	95.25	+7.29	62 s
Matching Network	89.29	95.93	+6.64	76 s
Relation Network	84.79	90.88	+6.09	75 s

<sub>Accuracy (%), mean ± std over 3 training seeds.</sub>

### Preliminary observations (single run, original notebooks — to be confirmed)

In the exploratory notebooks, Matching and Prototypical Networks were close (≈ 88–89% in 5-way 1-shot,
≈ 95–96% in 5-way 5-shot), the Relation Network was lower (≈ 85% / 91%), and the Siamese Network was
clearly the lowest (≈ 72% / 82%). Because those notebooks used slightly different evaluation protocols,
one seed, and different embedding sizes, treat this ordering as a hypothesis, not a conclusion.

## Limitations

* Omniglot is a relatively easy benchmark; conclusions may not transfer to mini-ImageNet or CUB.
* Short training budget (1000 episodes) with a randomly initialised ResNet18; rankings could change with
  longer training or a pre-trained backbone.
* The Siamese Network uses a different training signal (pairs) than the three episode-level methods, so its
  gap partly reflects the objective, not only the metric.
* Embedding sizes differ between methods (128-d vs 512-d) by design; see the ablation suggested in
  `docs/methodology.md`.

## Suggested extensions

1. Embedding-size ablation (all methods at 512-d).
2. Longer training and learning-rate schedules.
3. Cosine vs. Euclidean distance inside ProtoNet (isolates the metric alone).
4. Harder datasets (mini-ImageNet, CUB-200) and cross-domain evaluation.

## References

1. Koch, Zemel, Salakhutdinov. *Siamese Neural Networks for One-shot Image Recognition.* ICML Deep Learning Workshop, 2015.
2. Vinyals et al. *Matching Networks for One Shot Learning.* NeurIPS, 2016.
3. Snell, Swersky, Zemel. *Prototypical Networks for Few-shot Learning.* NeurIPS, 2017.
4. Sung et al. *Learning to Compare: Relation Network for Few-Shot Learning.* CVPR, 2018.
5. Lake, Salakhutdinov, Tenenbaum. *Human-level concept learning through probabilistic program induction* (Omniglot). Science, 2015.
6. Hadsell, Chopra, LeCun. *Dimensionality Reduction by Learning an Invariant Mapping* (contrastive loss). CVPR, 2006.
7. [`easyfsl`](https://github.com/sicara/easy-few-shot-learning) — Sicara, easy few-shot learning library.

## License

MIT — see [`LICENSE`](LICENSE).
