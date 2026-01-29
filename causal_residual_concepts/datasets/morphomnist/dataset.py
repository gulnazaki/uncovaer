import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms as tf
import numpy as np
import random

import sys
sys.path.append("../../")
from datasets.morphomnist import load_morphomnist_like
from datasets.morphomnist.config import ALL_CONCEPTS, TRAIN_VAL_SPLIT, SHORTCUT


def _worker_init_fn(worker_id, seed=42):
    """Seed each worker's RNG for deterministic data loading with num_workers > 0."""
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


class MorphoMNISTLike(Dataset):
    def __init__(self, attributes, split='train', transforms=None, binarize=True,
                 data_dir=None, test_ood=False, load_residuals=False, shortcuts=None, coefficients=None):

        self.has_valid_set = False

        if shortcuts is None:
            shortcuts = [SHORTCUT]

        if data_dir is None:
            data_dir = 'data_confounded'
            print(f"W: data_dir not provided, defaulting to {data_dir}")
        else:
            print(f"Loading from {data_dir}")
        self.root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_dir)

        self.train = split == 'train'
        self.transforms = transforms
        self.pad = tf.Pad(padding=2)

        test_prefix = 'test-ood' if test_ood else 'test-id'

        images, _, metrics_df = load_morphomnist_like(self.root_dir, self.train, columns=None, test_prefix=test_prefix)

        # Add channel dimension and normalize to [0,1]
        images = torch.as_tensor(images.copy(), dtype=torch.float32) / 255.0
        images = self.pad(images.unsqueeze(1))  # Add channel dimension (N, 1, H, W)

        self.images = images

        columns = ALL_CONCEPTS

        # Create binary or continuous concepts
        process_metric = (lambda col: (torch.as_tensor(metrics_df[col].values, dtype=torch.float32) > 0.5).float()) \
                 if binarize else \
                 (lambda col: torch.as_tensor(metrics_df[col].values, dtype=torch.float32))
        self.concepts_dict = {col: process_metric(col) for col in columns}

        # (N, num_concepts)
        self.concepts = torch.cat([self.concepts_dict[c].unsqueeze(1) for c in columns], dim=1)

        coeffs_np = np.array(coefficients)
        C = self.concepts[:, :len(coeffs_np)].numpy()
        p = (C @ coeffs_np).astype(float)
        y = np.random.binomial(n=1, p=p, size=C.shape[0]).astype(float)
        self.task = torch.as_tensor(y, dtype=torch.float32)
        
        self.attr = torch.cat([self.concepts, self.task.unsqueeze(1)], dim=1)
        self.task_id = self.attr.shape[1] - 1

        self.concept_ids = torch.tensor([columns.index(a) for a in attributes])
        self.shortcut_ids = torch.tensor([columns.index(a) for a in shortcuts])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, self.attr[idx]

    def get_indices(self):
        return {
            'concepts': self.concept_ids,
            'task': self.task_id,
            'shortcut': self.shortcut_ids,
        }

def get_dataloader(batch_size, split, attributes, transforms=None, binarize=True, test_ood=False, data_dir=None, shortcuts=None, coefficients=None, seed=42):
    data = MorphoMNISTLike(attributes, split=split.split('+')[0], transforms=transforms, binarize=binarize, test_ood=test_ood, data_dir=data_dir, shortcuts=shortcuts, coefficients=coefficients)
    if split == 'train':
        g = torch.Generator()
        g.manual_seed(seed)
        train_set, val_set = torch.utils.data.random_split(data, [TRAIN_VAL_SPLIT, 1 - TRAIN_VAL_SPLIT], generator=g)
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, generator=g, worker_init_fn=lambda wid: _worker_init_fn(wid, seed=seed))
        val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=lambda wid: _worker_init_fn(wid, seed=seed))
        return train_data_loader, val_data_loader, data.get_indices()
    elif split == 'test':
        test_data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=lambda wid: _worker_init_fn(wid, seed=seed))
        return test_data_loader, data.get_indices()
    elif split == 'train+val':
        train_val_data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=lambda wid: _worker_init_fn(wid, seed=seed))
        return train_val_data_loader, data.get_indices()
    else:
        raise NotImplementedError
