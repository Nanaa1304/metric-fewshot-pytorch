from typing import Optional

import torch.nn as nn
from torchvision import models


def resnet18_embedding(embedding_dim: Optional[int] = None) -> nn.Module:
    """ResNet18 (random init) producing a vector embedding.

    embedding_dim=None -> 512-d global-average-pool output (fc replaced by Flatten)
    embedding_dim=d    -> fc replaced by Linear(512, d)
    """
    net = models.resnet18(weights=None)
    in_features = net.fc.in_features  # 512
    net.fc = nn.Flatten() if embedding_dim is None else nn.Linear(in_features, embedding_dim)
    return net


def resnet18_feature_map() -> nn.Module:
    """ResNet18 without avgpool/fc: outputs a (512, H, W) feature map
    (512 x 3 x 3 for 84x84 inputs). Needed by the Relation Network."""
    net = models.resnet18(weights=None)
    return nn.Sequential(*list(net.children())[:-2])
