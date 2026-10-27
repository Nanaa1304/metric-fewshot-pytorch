# Methodology

## Research question

How do the four classical **metric-based** few-shot learners — Siamese Network, Prototypical Network,
Matching Network and Relation Network — compare when they share the same dataset, backbone,
training budget and evaluation episodes?

## Controlled factors (identical for all methods)

| Factor | Value |
|---|---|
| Dataset | Omniglot, torchvision official split: `background=True` (964 classes) for meta-training, `background=False` (659 unseen classes) for meta-testing |
| Input | grayscale → 3 channels, 84×84, normalised with mean = std = 0.5 |
| Backbone | ResNet18, random initialisation (no ImageNet pre-training) |
| Training | 1000 episodes, 5-way 1-shot, 5 queries per class |
| Optimiser | Adam, lr = 1e-4, weight decay = 1e-5 |
| Episode sampling | `easyfsl.TaskSampler`: classes drawn uniformly without replacement, then support/query images drawn uniformly without replacement inside each class |
| Test episodes | 600 episodes per setting, 15 queries per class, **same episodes for every method** (shared `episode_seed`) |
| Settings | 5-way 1-shot, 5-way 5-shot, 10-way 1-shot, 10-way 5-shot (models are trained once in 5-way 1-shot and tested on all four) |
| Seeds | 3 independent training runs per method (0, 1, 2) |

## Factors that differ by design (the object of the study)

| | Siamese | ProtoNet | Matching Net | Relation Net |
|---|---|---|---|---|
| Class representation | individual support embeddings | mean embedding (prototype) | all support embeddings, contextualised by a BiLSTM (FCE) | mean feature map (prototype) |
| Comparison function | Euclidean distance (fixed) | squared Euclidean distance (fixed) | cosine similarity + softmax attention (fixed) | CNN relation module (**learned**) |
| Training loss | contrastive loss on pairs (margin 1.0) | cross-entropy on −distance | NLL on attention-weighted label log-probabilities | MSE to one-hot target |
| Embedding | 128-d vector | 128-d vector | 512-d vector | 512×3×3 feature map |
| Implementation | from scratch | from scratch | `easyfsl.MatchingNetworks` | `easyfsl.RelationNetworks` |

> **Known confound.** Siamese/ProtoNet use a 128-d projection while Matching/Relation use the native
> 512-d ResNet18 output (as in the original notebooks). To separate the effect of the metric from the
> effect of the embedding size, set `methods.protonet.embedding_dim` and `methods.siamese.embedding_dim`
> to 512 in `configs/default.yaml` and re-run — a recommended ablation.

## Reported metrics

* **Accuracy** per setting: mean over the test episodes, then mean ± std across training seeds; the
  average episode-level 95% confidence half-width is also stored.
* **Macro precision / recall / F1** over all pooled query predictions.
* **Efficiency**: number of parameters, training wall-clock time, inference time per episode, peak GPU
  memory during training.
* **Paired bootstrap** of per-episode accuracy differences between every pair of methods (10 000
  resamples, 95% CI). Pairing is valid because all methods are evaluated on identical episodes.

## Differences between the original notebooks and this unified protocol

1. Evaluation episodes/queries were not identical across notebooks (600 episodes × 5 queries for
   Siamese and ProtoNet, 100 episodes × 15 queries for Matching and Relation). Now: 600 × 15 for all.
2. Test episodes are now identical across methods (seeded sampler) instead of independent draws.
3. Results were from a single seed; now 3 seeds with mean ± std.
4. Siamese notebook: an indentation error (`embedding1 = embeddings[indices1]`) was fixed, and the
   duplicated `evaluate_siamese` definition was removed.
5. ProtoNet notebook: the final summary cell used `precision_5shot` / `recall_5shot` / `f1_5shot`,
   which were never computed (NameError). Metrics are now computed for every setting by the engine.
6. ProtoNet notebook printed `len(dataset.get_labels())` as "number of classes", but that is the
   number of *images* (19 280 / 13 180); the number of classes is 964 / 659.
7. Relation Network: the 10-way settings had not been run in the notebook.

## Reproducibility

Seeds for Python, NumPy and PyTorch are fixed, cuDNN is set to deterministic mode, and the test
episode sampler is re-seeded before every evaluation. Exact bit-for-bit reproducibility across
different GPUs / PyTorch versions is not guaranteed.
