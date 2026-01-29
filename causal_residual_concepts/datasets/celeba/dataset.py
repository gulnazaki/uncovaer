from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose, ConvertImageDtype, RandomHorizontalFlip
import torchvision.models as models
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import os
import numpy as np
from .config import TASK, CAUSAL_CONCEPTS


def _worker_init_fn(worker_id, seed=42):
    """Seed each worker's RNG for deterministic data loading with num_workers > 0."""
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


MIN_MAX = {
    'image': [0.0, 255.0]
}


def apply_synthetic_label(
    attr: torch.Tensor,
    attr_names,
    task_name: str,
    causal_concepts,
    coefficients,
    seed: int = 42,
):
    """Replace the task column with a synthetic binary label.

    The label is generated via a logistic model depending on the
    concepts in ``causal_concepts`` and their interaction. For the
    current CelebA use case this is typically ["Young", "Male"], with
    coefficients provided in ``coefficients``.
    """

    for c in causal_concepts:
        if c not in attr_names:
            raise ValueError(f"{c} not found in CelebA attributes.")

    # Base intercept; keep default if not specified.
    b0 = coefficients.get("base", 0.5)

    # Probabilities
    p = torch.full((attr.shape[0],), b0, dtype=torch.float32)

    # Add main effects for each causal concept.
    for c in causal_concepts:
        idx = attr_names.index(c)
        beta = coefficients.get(c, 0.0)
        p = p + beta * attr[:, idx].float()

    # Add pairwise interaction terms of the form "A_B".
    for i in range(len(causal_concepts)):
        for j in range(i + 1, len(causal_concepts)):
            c1, c2 = causal_concepts[i], causal_concepts[j]
            key = f"{c1}_{c2}"
            if key in coefficients:
                beta_ij = coefficients[key]
                idx1 = attr_names.index(c1)
                idx2 = attr_names.index(c2)
                p = p + beta_ij * (
                    attr[:, idx1].float() * attr[:, idx2].float()
                )

    g = torch.Generator()
    g.manual_seed(seed)
    y_synth = torch.bernoulli(p, generator=g).to(attr.dtype)

    task_id = attr_names.index(task_name)
    attr[:, task_id] = y_synth

    return attr, task_id

def load_data(data_dir, split, dimension=64):
    transforms = Compose([CenterCrop(128), Resize((dimension, dimension)), ToTensor(), ConvertImageDtype(dtype=torch.float32),])
    data = CelebA(root=data_dir, split=split, transform=transforms, download=False)
    return data

def load_embeddings(data_dir, split, flip=False):
    emb_suffix = "dinov2_flipped_embeddings.pt" if flip else "dinov2_embeddings.pt"
    if split == 'all':
        embeddings = []
        for split in ['train', 'valid', 'test']:
            emb_path = os.path.join(data_dir, "celeba", f"{split}_{emb_suffix}")
            embeddings.append(torch.load(emb_path))
        embeddings = torch.cat(embeddings)
    else:
        emb_path = emb_path = os.path.join(data_dir, "celeba", f"{split}_{emb_suffix}")
        embeddings = torch.load(emb_path)

    return embeddings

def unnormalize(value, name):
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value.to(torch.uint8)

class CelebADataset(Dataset):
    def __init__(self, attributes, split='train', transforms=None, data_dir='/storage', use_dinov2_embeddings=False, use_cached_images=False, balance=False, task=None, only_attr=False, dimension=64, coefficients=None, shortcuts=None, seed=42):
        super().__init__()
        self.use_dinov2_embeddings = use_dinov2_embeddings
        self.data_dir = data_dir
        self.transforms = transforms
        self.concept_names = attributes if len(attributes) > 0 else [TASK]
        self.task_name = TASK if task is None else task
        self.coefficients = coefficients

        self.data = load_data(data_dir, split, dimension=dimension)

        if self.use_dinov2_embeddings:
            self.embeddings = load_embeddings(data_dir, split, flip=False)
            if self.transforms is not None and type(self.transforms) == RandomHorizontalFlip:
                self.flipped_embeddings = load_embeddings(data_dir, split, flip=True)

        self.attr_names = self.data.attr_names
        self.attr = self.data.attr.clone()

        # Optionally replace the task label with a synthetic binary label
        # driven by CAUSAL_CONCEPTS and COEFFICIENTS.
        if self.coefficients is not None:
            self.attr, self.task_id = apply_synthetic_label(
                self.attr,
                self.attr_names,
                self.task_name,
                causal_concepts=CAUSAL_CONCEPTS,
                coefficients=coefficients,
                seed=seed,
            )
        else:
            self.task_id = self.attr_names.index(self.task_name)

        self.concept_ids = torch.tensor([self.attr_names.index(c) for c in self.concept_names])

        self.active_indices = None

        self.only_attr = only_attr
        self.use_cached_images = use_cached_images
        self.cached_images = None
        if self.use_cached_images:
            cache_path = os.path.join(data_dir, "celeba", f"{split}_images_64x64.pt")
            if os.path.isfile(cache_path):
                self.cached_images = torch.load(cache_path)
        
        self.shortcuts = shortcuts
        self.shortcut_ids = torch.tensor([self.attr_names.index(s) for s in shortcuts]) if shortcuts is not None else None


    def __len__(self):
        return len(self.active_indices) if self.active_indices is not None else len(self.data)

    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        
        if self.only_attr:
            return 0, self.attr[actual_idx].float()
        
        if self.use_dinov2_embeddings:
            if self.transforms is None:
                img = self.embeddings[actual_idx]
            elif isinstance(self.transforms, RandomHorizontalFlip):
                img = self.flipped_embeddings[actual_idx] if torch.rand(1).item() > self.transforms.p else self.embeddings[actual_idx]
            else:
                raise RuntimeError("Only RandomHorizontalFlip is supported for DINOv2 embeddings.")
        else:
            if self.cached_images is not None:
                img = self.cached_images[actual_idx]
                if isinstance(self.transforms, RandomHorizontalFlip) and torch.rand(1).item() < self.transforms.p:
                    img = torch.flip(img, dims=[2])  # horizontal flip on W (C,H,W)
            else:
                img = self.data[actual_idx][0]
                img = self.transforms(img) if self.transforms is not None else img

        return img, self.attr[actual_idx].float()

    def get_indices(self):
        return {
            'concepts': self.concept_ids,
            'task': self.task_id,
            'shortcut': self.shortcut_ids
        }


def get_dataloader(batch_size, split, attributes, transforms=None, data_dir='/storage', use_dinov2_embeddings=False, only_attr=False, use_cached_images=False, balance=False, shortcuts=None, task=None, coefficients=None, seed=42):
    data = CelebADataset(attributes, split=split, transforms=transforms, data_dir=data_dir,
                         use_dinov2_embeddings=use_dinov2_embeddings, balance=balance,
                         use_cached_images=use_cached_images, task=task, only_attr=only_attr,
                         coefficients=coefficients, shortcuts=shortcuts, seed=seed)

    shuffle = True if split == 'train' else False
    # Create a seeded generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=8,
                                              pin_memory=True, generator=g if shuffle else None, worker_init_fn=lambda wid: _worker_init_fn(wid, seed=seed))
    return data_loader, data.get_indices()
