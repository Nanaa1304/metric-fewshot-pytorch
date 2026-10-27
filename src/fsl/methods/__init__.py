METHOD_NAMES = ["siamese", "protonet", "matching", "relation"]


def build_method(name: str, method_cfg: dict):
    """Instantiate a method from its name and its section of the YAML config.
    Imports are lazy so that a missing optional dependency only affects that method."""
    cfg = dict(method_cfg or {})
    if name == "protonet":
        from .protonet import ProtoNet
        return ProtoNet(**cfg)
    if name == "siamese":
        from .siamese import SiameseNet
        return SiameseNet(**cfg)
    if name == "matching":
        from .matching import MatchingNet
        return MatchingNet(**cfg)
    if name == "relation":
        from .relation import RelationNet
        return RelationNet(**cfg)
    raise ValueError(f"Unknown method '{name}'. Choose from {METHOD_NAMES}.")
