import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from ..backbones import resnet18_embedding
from .base import FewShotMethod


class ContrastiveLoss(torch.nn.Module):
    """L = 1/2 [ y D^2 + (1 - y) max(0, m - D)^2 ]  (Hadsell et al., 2006)."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        positive = label * distance.pow(2)
        negative = (1 - label) * torch.clamp(self.margin - distance, min=0.0).pow(2)
        return 0.5 * (positive + negative).mean()


def create_pairs(labels, pairs_per_class: int):
    """Balanced pairs from one episode: (idx1, idx2, 1=same class / 0=different)."""
    by_class = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        by_class[label].append(index)
    classes = list(by_class.keys())

    positives = []
    for c in classes:
        for _ in range(pairs_per_class):
            i, j = random.sample(by_class[c], 2)
            positives.append((i, j, 1))

    negatives = []
    for _ in range(len(positives)):
        c1, c2 = random.sample(classes, 2)
        negatives.append((random.choice(by_class[c1]), random.choice(by_class[c2]), 0))

    pairs = positives + negatives
    random.shuffle(pairs)
    return pairs


class SiameseNet(FewShotMethod):
    """Siamese Network with a shared ResNet18 encoder, Euclidean distance and
    contrastive loss trained on pairs built inside each episode.

    Inference: a query is assigned to the class with the smallest MEAN Euclidean
    distance to that class's support embeddings.
    """

    name = "siamese"

    def __init__(self, embedding_dim: int = 128, margin: float = 1.0, pairs_per_class: int = 5):
        super().__init__()
        self.encoder = resnet18_embedding(embedding_dim)
        self.criterion = ContrastiveLoss(margin)
        self.pairs_per_class = pairs_per_class

    @staticmethod
    def _class_scores(z_support, support_labels, z_query, n_way):
        distances = torch.cdist(z_query, z_support, p=2)               # (Q, S)
        one_hot = F.one_hot(support_labels, n_way).float()             # (S, N)
        mean_distance = distances @ one_hot / one_hot.sum(dim=0)       # (Q, N)
        return -mean_distance

    def episode_loss(self, support_images, support_labels, query_images, query_labels, n_way):
        images = torch.cat([support_images, query_images], dim=0)
        labels = torch.cat([support_labels, query_labels], dim=0)
        embeddings = self.encoder(images)

        pairs = create_pairs(labels.cpu(), self.pairs_per_class)
        idx1 = torch.tensor([p[0] for p in pairs], device=embeddings.device)
        idx2 = torch.tensor([p[1] for p in pairs], device=embeddings.device)
        targets = torch.tensor([p[2] for p in pairs], dtype=torch.float32, device=embeddings.device)

        distances = F.pairwise_distance(embeddings[idx1], embeddings[idx2])
        loss = self.criterion(distances, targets)

        # Episode accuracy for monitoring only (no gradient involved).
        n_support = len(support_labels)
        scores = self._class_scores(
            embeddings[:n_support].detach(), support_labels, embeddings[n_support:].detach(), n_way
        )
        return loss, scores

    def predict_scores(self, support_images, support_labels, query_images):
        n_way = int(support_labels.max().item()) + 1
        return self._class_scores(
            self.encoder(support_images), support_labels, self.encoder(query_images), n_way
        )
