from .. import compat  # noqa: F401

import torch.nn.functional as F
from easyfsl.methods import MatchingNetworks

from ..backbones import resnet18_embedding
from .base import FewShotMethod


class MatchingNet(FewShotMethod):
    """Matching Networks with Full Context Embeddings (Vinyals et al., 2016),
    implemented by easyfsl: bidirectional-LSTM support context, LSTM-with-attention
    query embedding, cosine attention, prediction = attention-weighted support labels.
    The model outputs log-probabilities -> NLL loss.
    """

    name = "matching"

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        backbone = resnet18_embedding(None)  # 512-d (global average pool)
        self.model = MatchingNetworks(backbone, feature_dimension=embedding_dim)

    def predict_scores(self, support_images, support_labels, query_images):
        self.model.process_support_set(support_images, support_labels)
        return self.model(query_images)

    def episode_loss(self, support_images, support_labels, query_images, query_labels, n_way):
        log_probabilities = self.predict_scores(support_images, support_labels, query_images)
        return F.nll_loss(log_probabilities, query_labels), log_probabilities
