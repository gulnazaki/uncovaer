#!/usr/bin/env python
import os
import sys
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.concept_classifiers import DinoV2
from datasets.celeba.dataset import CelebADataset


def precompute_features(data_dir, splits, batch_size, workers, mode, fp16, force):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    need_emb = mode in ("both", "embeddings")
    need_img = mode in ("both", "images")

    model = None
    if need_emb:
        model = DinoV2(num_concepts=0).to(device)
        model.eval()

    os.makedirs(os.path.join(data_dir, "embeddings"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    need_emb = mode in ("both", "embeddings")
    need_img = mode in ("both", "images")

    model = None
    if need_emb:
        model = DinoV2(num_concepts=0).to(device)
        model.eval()

    os.makedirs(os.path.join(data_dir, "embeddings"), exist_ok=True)

    for split in splits:
        img_cache = os.path.join(data_dir, "celeba", f"{split}_images_64x64.pt")
        emb_path = os.path.join(data_dir, "celeba", f"{split}_dinov2_embeddings.pt")
        emb_flip_path = os.path.join(data_dir, "celeba", f"{split}_dinov2_flipped_embeddings.pt")

        do_img = need_img and (force or not os.path.isfile(img_cache))
        do_emb = need_emb and (force or not (os.path.isfile(emb_path) and os.path.isfile(emb_flip_path)))

        if not (do_img or do_emb):
            print(f"[{split}] All requested assets already exist.")
            continue


        imgs_buf = None
        if do_img:
            dataset = CelebADataset(attributes=[], split=split, data_dir=data_dir, dimension=64, use_dinov2_embeddings=False)
            img_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
            N = len(dataset)
            dtype = torch.float16 if fp16 else torch.float32
            imgs_buf = torch.empty((N, 3, 64, 64), dtype=dtype)

        feats_buf = feats_flip_buf = None
        if do_emb:
            dataset = CelebADataset(attributes=[], split=split, data_dir=data_dir, dimension=70, use_dinov2_embeddings=False)
            emb_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
            N = len(dataset)
            # Determine embedding dim from first batch
            with torch.no_grad():
                first_images, _ = next(iter(emb_loader))
                f_dim = model(first_images.to(device))[1].shape[1]
            dtype = torch.float16 if fp16 else torch.float32
            feats_buf = torch.empty((N, f_dim), dtype=dtype)
            feats_flip_buf = torch.empty((N, f_dim), dtype=dtype)

        hflip = transforms.functional.hflip
        

        with torch.no_grad():
            offset = 0
            if do_img:
                for images, _ in tqdm(img_loader, desc=f"Precompute {split}"):
                    b = images.shape[0]
                    batch = images.half() if fp16 else images
                    imgs_buf[offset:offset+b] = batch
                    offset += b
            
            offset = 0
            if do_emb:
                for images, _ in tqdm(emb_loader, desc=f"Precompute embeddings {split}"):
                    b = images.shape[0]
                    x = images.to(device)
                    x_flip = hflip(x)
                    _, f = model(x)
                    _, f_flip = model(x_flip)
                    f = f.half().cpu() if fp16 else f.cpu()
                    f_flip = f_flip.half().cpu() if fp16 else f_flip.cpu()
                    feats_buf[offset:offset+b] = f
                    feats_flip_buf[offset:offset+b] = f_flip
                    offset += b

        if do_img:
            torch.save(imgs_buf, img_cache)
            print(f"[{split}] Saved images tensor: {img_cache} shape={tuple(imgs_buf.shape)} dtype={imgs_buf.dtype}")

        if do_emb:
            torch.save(feats_buf, emb_path)
            torch.save(feats_flip_buf, emb_flip_path)
            print(f"[{split}] Saved embeddings: {emb_path} shape={tuple(feats_buf.shape)} dtype={feats_buf.dtype}")
            print(f"[{split}] Saved flipped embeddings: {emb_flip_path} shape={tuple(feats_flip_buf.shape)} dtype={feats_flip_buf.dtype}")



def file_exists(p): return os.path.isfile(p)

def need_images(data_dir, splits):
    return any(not file_exists(os.path.join(data_dir, "celeba", f"{s}_images_64x64.pt")) for s in splits)

def need_embeddings(data_dir, splits):
    emb_dir = os.path.join(data_dir, "celeba")
    return any(
        not file_exists(os.path.join(emb_dir, f"{s}_dinov2_embeddings.pt")) or
        not file_exists(os.path.join(emb_dir, f"{s}_dinov2_flipped_embeddings.pt"))
        for s in splits
    )

def precompute_if_needed(data_dir, want_images, want_embeddings, force=False, fp16=True, batch_size=128, workers=8, splits=[]):
    imgs_missing = want_images and (force or need_images(data_dir, splits))
    emb_missing = want_embeddings and (force or need_embeddings(data_dir, splits))
    if not (imgs_missing or emb_missing):
        print("Precompute: nothing to do.")
        return
    mode = "both" if (imgs_missing and emb_missing) else ("images" if imgs_missing else "embeddings")

    precompute_features(
        data_dir=data_dir,
        splits=splits,
        batch_size=batch_size,
        workers=workers,
        mode=mode,
        fp16=fp16,
        force=force
    )
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--force", action="store_true", help="Force regeneration")
    ap.add_argument("--fp16", action="store_true", help="Save as fp16")
    args = ap.parse_args()

    SPLITS = ["train", "valid", "test"]

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg.get("data_dir", "/netdisk/data/CV/")
    use_cached = bool(cfg.get("use_cached_images", False))
    use_dino = bool(cfg.get("use_dinov2_embeddings", False))
    batch_size = cfg.get("batch_size", 128)
    workers = cfg.get("num_workers", 8)
    splits = SPLITS

    precompute_if_needed(
        data_dir=data_dir,
        want_images=use_cached,
        want_embeddings=use_dino,
        force=args.force,
        fp16=args.fp16,
        batch_size=batch_size,
        workers=workers,
        splits=splits
    )

if __name__ == "__main__":
    main()