from . import compat  # noqa: F401  (must come before easyfsl imports)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from easyfsl.datasets import WrapFewShotDataset
from easyfsl.samplers import TaskSampler


def build_transform(image_size: int = 84):
    # Omniglot is grayscale; ResNet18 expects 3 channels.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_omniglot(root: str = "./data", image_size: int = 84):
    """Official torchvision split:
    background=True  -> 964 classes (meta-training)
    background=False -> 659 unseen classes (meta-test)
    """
    transform = build_transform(image_size)
    raw_train = datasets.Omniglot(root=root, background=True, download=True, transform=transform)
    raw_test = datasets.Omniglot(root=root, background=False, download=True, transform=transform)
    return WrapFewShotDataset(raw_train), WrapFewShotDataset(raw_test)


def episodic_loader(dataset, n_way: int, k_shot: int, q_query: int, n_tasks: int, num_workers: int = 2):
    """Each batch yielded by the loader is one full episode:
    (support_images, support_labels, query_images, query_labels, true_class_ids).
    Labels inside an episode are re-indexed to 0..n_way-1.
    """
    sampler = TaskSampler(dataset, n_way=n_way, n_shot=k_shot, n_query=q_query, n_tasks=n_tasks)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=sampler.episodic_collate_fn,
    )
