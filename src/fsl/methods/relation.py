from .. import compat  # noqa: F401

import torch.nn.functional as F
from easyfsl.methods import RelationNetworks

from ..backbones import resnet18_feature_map
from .base import FewShotMethod


class RelationNet(FewShotMethod):
    """Relation Network (Sung et al., 2018), implemented by easyfsl.

    The comparison function is LEARNED: a small CNN (relation module) takes the
    channel-wise concatenation of a query feature map and a class prototype map and
    outputs a relation score in [0, 1]. Trained by regression (MSE) towards the
    one-hot target.
    """

    name = "relation"

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        backbone = resnet18_feature_map()
        self.model = RelationNetworks(backbone, feature_dimension=embedding_dim)

    def predict_scores(self, support_images, support_labels, query_images):
        self.model.process_support_set(support_images, support_labels)
        return self.model(query_images)

    def episode_loss(self, support_images, support_labels, query_images, query_labels, n_way):
        scores = self.predict_scores(support_images, support_labels, query_images)
        targets = F.one_hot(query_labels, n_way).float()
        return F.mse_loss(scores, targets), scores
