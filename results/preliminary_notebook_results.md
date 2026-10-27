# Preliminary results (original notebooks)

Single training run (one seed), 1000 episodes of 5-way 1-shot, ResNet18, Omniglot (84x84),
evaluated on the 659 unseen classes. **These numbers come from the four exploratory notebooks in
`notebooks/original/`, which did not share exactly the same evaluation protocol**
(see `docs/methodology.md`), so they must not be used for a strict ranking.
The final, protocol-controlled numbers are produced by `scripts/run_all.sh`.

| Method | 5-way 1-shot | 5-way 5-shot | 10-way 1-shot | 10-way 5-shot | Eval protocol in notebook |
|---|---|---|---|---|---|
| Siamese (Euclidean + contrastive) | 72.19 | 81.91 | not run | not run | 600 episodes, 5 queries/class |
| ProtoNet (squared Euclidean) | 87.96 | 95.25 | not run | not run | 600 episodes, 5 queries/class |
| Matching Net (cosine + FCE) | 89.29 ± 8.68 | 95.93 ± 4.03 | 82.63 ± 7.69 | 91.79 ± 4.00 | 100 episodes, 15 queries/class (± = std over episodes) |
| Relation Net (learned) | 84.79 ± 9.85 | 90.88 ± 5.67 | not run | not run | 100 episodes, 15 queries/class (± = std over episodes) |
