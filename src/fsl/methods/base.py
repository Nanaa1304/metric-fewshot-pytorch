import torch.nn as nn


class FewShotMethod(nn.Module):
    """Common interface so every method is trained/evaluated by the same engine.

    Scores returned by both functions have shape (n_query, n_way); a HIGHER score
    means "more likely", and the predicted class is ``scores.argmax(dim=1)``.
    Labels inside an episode are always 0..n_way-1 (re-indexed by the sampler).
    """

    name = "base"

    def episode_loss(self, support_images, support_labels, query_images, query_labels, n_way):
        """Training step on one episode -> (loss, scores)."""
        raise NotImplementedError

    def predict_scores(self, support_images, support_labels, query_images):
        """Inference on one episode -> scores."""
        raise NotImplementedError
