import torch
import torch.nn.functional as F

from ..backbones import resnet18_embedding
from .base import FewShotMethod


class ProtoNet(FewShotMethod):
    """Prototypical Network (Snell et al., 2017).

    Prototype = mean support embedding of each class; query classified by the
    SQUARED EUCLIDEAN distance to each prototype; loss = cross-entropy on -distance.
    """

    name = "protonet"

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = resnet18_embedding(embedding_dim)

    @staticmethod
    def compute_prototypes(support_embeddings, support_labels, n_way):
        one_hot = F.one_hot(support_labels, n_way).float()            # (S, N)
        counts = one_hot.sum(dim=0).unsqueeze(1)                       # (N, 1)
        return one_hot.t() @ support_embeddings / counts               # (N, D)

    def _scores(self, support_images, support_labels, query_images, n_way):
        z_support = self.encoder(support_images)
        z_query = self.encoder(query_images)
        prototypes = self.compute_prototypes(z_support, support_labels, n_way)
        return -(torch.cdist(z_query, prototypes, p=2) ** 2)

    def episode_loss(self, support_images, support_labels, query_images, query_labels, n_way):
        scores = self._scores(support_images, support_labels, query_images, n_way)
        return F.cross_entropy(scores, query_labels), scores

    def predict_scores(self, support_images, support_labels, query_images):
        n_way = int(support_labels.max().item()) + 1
        return self._scores(support_images, support_labels, query_images, n_way)
