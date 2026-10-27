"""Training and evaluation loops shared by all methods (identical protocol)."""
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from .data import episodic_loader
from .utils import mean_ci95


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train(method, loader, optimizer, device, n_way: int, log_every: int = 100):
    method.train()
    losses, accuracies = [], []
    for episode, (sx, sy, qx, qy, _) in enumerate(loader, start=1):
        sx, sy = sx.to(device, non_blocking=True), sy.to(device, non_blocking=True)
        qx, qy = qx.to(device, non_blocking=True), qy.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, scores = method.episode_loss(sx, sy, qx, qy, n_way)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append((scores.argmax(dim=1) == qy).float().mean().item())

        if episode == 1 or episode % log_every == 0:
            recent = slice(-log_every, None)
            print(f"  episode {episode:5d} | loss {np.mean(losses[recent]):.4f} "
                  f"| acc {100 * np.mean(accuracies[recent]):.2f}%")
    return {"loss": losses, "acc": accuracies}


@torch.no_grad()
def evaluate(method, dataset, n_way, k_shot, q_query, n_tasks, seed, device, num_workers=2):
    """Evaluate on `n_tasks` episodes. Because the episode sampler is seeded with
    `seed`, every method sees EXACTLY the same test episodes (paired comparison)."""
    loader = episodic_loader(dataset, n_way, k_shot, q_query, n_tasks, num_workers)
    method.eval()

    episode_acc, y_true, y_pred = [], [], []
    forward_seconds = 0.0

    random.seed(seed)  # TaskSampler draws from the global `random` module
    for sx, sy, qx, qy, _ in loader:
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        _sync(device)
        start = time.perf_counter()
        scores = method.predict_scores(sx, sy, qx)
        _sync(device)
        forward_seconds += time.perf_counter() - start

        predictions = scores.argmax(dim=1)
        episode_acc.append((predictions == qy).float().mean().item())
        y_true.extend(qy.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    mean, ci = mean_ci95(episode_acc)
    return {
        "n_way": n_way,
        "k_shot": k_shot,
        "q_query": q_query,
        "n_episodes": n_tasks,
        "accuracy": mean,
        "ci95": ci,
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ms_per_episode": 1000.0 * forward_seconds / n_tasks,
        "episode_accuracies": episode_acc,
    }
