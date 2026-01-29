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
from .config import ALL_ATTRIBUTES, TASK, BALANCE, SHORTCUT, CAUSAL_CONCEPTS, COEFFICIENTS, COEFFICIENTS_OOD
from .utils import top_confounders


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

        # correct_indices = torch.from_numpy(np.load(f'/home/ubuntu/celeba_experiments/smiling_dino_emb/{split}_correct_indices.npy'))
        # self.filter(correct_indices)

        # if balance:
        #     self.balance_dataset()

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

    def get_attribute_class_ratios(self):
        ones = (self.attr[:, self.concept_ids] == 1).sum(dim=0)
        ratios = len(self) / ones
        return ratios

    # def filter(self, indices):
    #     if self.active_indices is None:
    #         print(f"Filtering {len(indices)} out of {len(self.data)}")
    #         self.active_indices = indices
    #     else:
    #         a_cat_b, counts = torch.cat([self.active_indices, indices]).unique(return_counts=True)
    #         intersection = a_cat_b[torch.where(counts.gt(1))]
    #         print(f"Filtering {len(intersection)} out of {len(self.active_indices)}")
    #         self.active_indices = intersection

    # def apply_shortcut_filter(self, shortcut_attr=SHORTCUT, discard_for=0, keep_ratio=0.0):
    #     if shortcut_attr not in self.attr_names:
    #         raise ValueError(f"{shortcut_attr} not found in attribute names.")

    #     shortcut_id = self.attr_names.index(shortcut_attr)
    #     shortcut = torch.as_tensor(self.attr[:, shortcut_id], dtype=torch.float32)

    #     # Identify samples matching the shortcut+task condition to discard
    #     to_discard = ((shortcut == 1) & (self.attr[:, self.task_id] == discard_for))

    #     # Indices to keep from the "discard group"
    #     discard_indices = to_discard.nonzero(as_tuple=True)[0]
    #     num_to_keep = int(len(discard_indices) * keep_ratio)
    #     g = torch.Generator()
    #     g.manual_seed(SEED)
    #     keep_from_discard = discard_indices[torch.randperm(len(discard_indices), generator=g)[:num_to_keep]]

    #     # Indices to always keep (not matching the discard condition)
    #     to_keep = (~to_discard).nonzero(as_tuple=True)[0]

    #     # Combine indices
    #     final_keep_indices = torch.cat([to_keep, keep_from_discard]).sort().values

    #     self.filter(final_keep_indices)

    # def balance_dataset(self):
    #     balance_id = self.attr_names.index(BALANCE)
    #     balance_attr = torch.as_tensor(self.attr[:, balance_id], dtype=torch.float32).unsqueeze(1)

    #     task_balance = torch.cat([self.attr[:, self.task_id], balance_attr], dim=1)
    #     unique_combinations, inverse_indices = torch.unique(task_balance, dim=0, return_inverse=True)

    #     print(f"Number of ({TASK}, {BALANCE}) combinations: {len(unique_combinations)}")

    #     combo_counts = {}
    #     indices_per_combo = []
    #     min_count = float('inf')

    #     for combo_idx, combo in enumerate(unique_combinations):
    #         indices = (inverse_indices == combo_idx).nonzero(as_tuple=True)[0]
    #         indices_per_combo.append(indices)

    #         label_str = f"{int(combo[0].item())}_{int(combo[1].item())}"
    #         count = len(indices)
    #         combo_counts[label_str] = count

    #         min_count = min(min_count, count)

    #     print("Sample counts per (TASK, BALANCE) combination:")
    #     for label, count in combo_counts.items():
    #         print(f"  {label}: {count}")

    #     balanced_indices = torch.cat([indices[:min_count] for indices in indices_per_combo], dim=0)
    #     g = torch.Generator()
    #     g.manual_seed(SEED)
    #     shuffled = balanced_indices[torch.randperm(len(balanced_indices), generator=g)]

    #     print(f"Selecting {len(shuffled)} out of {len(self.data)} samples")

    #     self.filter(shuffled)

    # def select_only_attribute(self, attr):
    #     if attr not in self.attr_names:
    #         raise ValueError(f"{attr} not found in attribute names.")

    #     shortcut_id = self.attr_names.index(attr)
    #     shortcut = torch.as_tensor(self.attr[:, shortcut_id], dtype=torch.float32).unsqueeze(1)

    #     to_keep = (shortcut == 1).squeeze().nonzero(as_tuple=True)[0]

    #     self.filter(to_keep)

# class CelebADatasetBalancedRace(Dataset):
#     """
#     CelebA dataset with race predictions from FairFace, subsampled
#     so that number of non-white = number of white images.
#     """
#     def __init__(self, data_dir, split='train', device='cuda', seed=42):
#         super().__init__()

#         # Load CelebA
#         self.transforms = Compose([
#             CenterCrop(128),
#             Resize((64, 64)),
#             ToTensor(),
#             ConvertImageDtype(dtype=torch.float32)
#         ])
#         self.data = CelebA(root=data_dir, split=split, transform=self.transforms, download=False)
#         self.attr = self.data.attr.float()
#         self.attr_names = self.data.attr_names

#         # Load FairFace model
#         self.model = models.resnet34(pretrained=True)
#         self.model.fc = nn.Linear(self.model.fc.in_features, 4)  # 4-race classes
#         self.model.load_state_dict(torch.load('res34_fair_align_multi_4_20190809.pt', map_location=device))
#         self.model = self.model.to(device)
#         self.model.eval()

#         # Compute race predictions
#         self.race_preds = self.compute_race()

#         # Balance dataset
#         self.active_indices = self.balance_dataset()

#     def compute_race(self):
#         race_preds = []
#         preprocess = Compose([
#             Resize((224, 224)),
#             ToTensor()
#         ])
#         with torch.no_grad():
#             for idx in tqdm(range(len(self.data)), desc="Predicting race"):
#                 img, _ = self.data[idx]
#                 x = preprocess(img).unsqueeze(0).to(self.device)
#                 logits = self.model(x)
#                 race = torch.argmax(logits, dim=1).item()  # 0=White, 1=Black, 2=Asian, 3=Indian
#                 race_preds.append(race)
#         return torch.tensor(race_preds)

#     def balance_dataset(self):
#         # indices for white and non-white
#         white_idx = (self.race_preds == 0).nonzero(as_tuple=True)[0]
#         non_white_idx = (self.race_preds != 0).nonzero(as_tuple=True)[0]

#         # subsample the larger group
#         n = min(len(white_idx), len(non_white_idx))
#         white_sample = random.sample(white_idx.tolist(), n)
#         non_white_sample = random.sample(non_white_idx.tolist(), n)

#         balanced_indices = white_sample + non_white_sample
#         random.shuffle(balanced_indices)
#         return balanced_indices

#     def __len__(self):
#         return len(self.active_indices)

#     def __getitem__(self, idx):
#         actual_idx = self.active_indices[idx]
#         img, attr = self.data[actual_idx]
#         race = self.race_preds[actual_idx]
#         return img, attr, race

#     def get_indices(self):
#         return {
#             'concepts': self.concept_ids,
#             'task': self.task_id,
#             'shortcut': self.shortcut_ids
#         }


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
