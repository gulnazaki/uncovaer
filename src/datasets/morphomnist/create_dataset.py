import numpy as np
import pandas as pd
import torch
from torch.nn.functional import sigmoid
from tqdm import tqdm
import sys
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from datasets.morphomnist import load_morphomnist_like, save_morphomnist_like
from datasets.morphomnist.transforms import SetThickness, SetSlant, SetWidth, ImageMorphology, get_intensity


def sample_conditional_gaussian(n_samples, seed=None, confounders=None, confounded_prob=0.0):
    if seed is not None:
        np.random.seed(seed)

    labels = np.random.randint(0, 2, size=n_samples)
    if confounders is not None:
        p_copy = confounded_prob
        change_ids = np.random.rand(n_samples) < p_copy
        if len(confounders) == 2:
            labels[change_ids] = np.logical_or(confounders[0][change_ids], confounders[1][change_ids]).astype(int).copy()
        elif len(confounders) == 1:
            labels[change_ids] = confounders[0][change_ids].copy()
        else:
            raise RuntimeError(f"Only 1 or 2 confounder implemented, got {len(confounders)}")

    values = np.empty(n_samples)

    values[labels == 0] = np.clip(
        np.random.normal(loc=0.25, scale=0.01, size=(labels == 0).sum()),
        0.05, 0.45
    )
    values[labels == 1] = np.clip(
        np.random.normal(loc=0.75, scale=0.01, size=(labels == 1).sum()),
        0.55, 0.95
    )

    return labels, values


def scale_features(thickness, slant, width, intensity):
    t = thickness * 6                     # [0, 6]
    s = slant * 180 - 90                  # [-90, 90]
    w = width * 15 + 10                   # [10, 25]
    i = intensity * 255                   # [0, 255]
    return t, s, w, i


def generate_image(image, thickness, intensity, width, slant):
    morph = ImageMorphology(image, scale=16)
    tmp_img = np.float32(SetThickness(thickness)(morph))
    tmp_img = np.float32(SetSlant(np.deg2rad(slant))(ImageMorphology(tmp_img, scale=1)))
    tmp_img = np.float32(SetWidth(width * 16)(ImageMorphology(tmp_img, scale=1)))
    tmp_img = morph.downscale(tmp_img)

    avg_intensity = get_intensity(tmp_img)
    mult = intensity / avg_intensity if avg_intensity > 0 else 1.0
    return np.clip(tmp_img * mult, 0, 255)


def process_one(args):
    img, t, i, w, s = args
    return generate_image(img, t, i, w, s)


def chunked_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]


def process_batch(batch):
    out = []
    for img, t, inten, w, s in batch:
        out.append(generate_image(img, t, inten, w, s))
    return out


def gen_dataset(
    data_dir='mnist',
    out_dir='data',
    digit_class=0,
    train=True,
    mode='id'
):
    images_, labels = load_morphomnist_like(data_dir, train=train, no_metrics=True, test_prefix='t10k')

    if digit_class is not None:
        mask = labels == digit_class
        images_ = images_[mask]
        labels = labels[mask]

    n_samples = len(images_)
    
    thickness_labels, thickness = sample_conditional_gaussian(n_samples)

    slant_labels, slant = sample_conditional_gaussian(n_samples, confounders=[thickness_labels], confounded_prob=(0.9 if mode == 'id' else 0.6))

    intensity_labels, intensity = sample_conditional_gaussian(n_samples, confounders=[thickness_labels], confounded_prob=(0.9 if mode == 'id' else 0.6))

    width_labels, width = sample_conditional_gaussian(n_samples)
    
    metrics = pd.DataFrame({
        'thickness': thickness,
        'slant': slant,
        'width': width,
        'intensity': intensity,
        'thickness_labels': thickness_labels,
        'slant_labels': slant_labels,
        'width_labels': width_labels,
        'intensity_labels': intensity_labels,
    })

    thickness_scaled, slant_scaled, width_scaled, intensity_scaled = scale_features(thickness, slant, width, intensity)

    args = [
        (images_[n], thickness_scaled[n], intensity_scaled[n], width_scaled[n], slant_scaled[n])
        for n in range(n_samples)
    ]

    # Process items in batches to reduce inter-process pickling/IPC overhead
    batch_size = 128
    batches = list(chunked_iterable(args, batch_size))

    import multiprocessing
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results_batches = list(tqdm(executor.map(process_batch, batches), total=len(batches)))

    # flatten results and stack
    results = [img for batch in results_batches for img in batch]
    images = np.stack(results)

    prefix = f"{'train' if train else f'test-{mode}'}"
    save_morphomnist_like(images, labels, metrics, out_dir, prefix=prefix)

    return metrics, labels


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'mnist')
    out_dir = os.path.join(current_dir, 'data_confounded_is_by_t')
    digit_classes = [0, 1, 2, 3, 4]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    def process_digit(digit):
        out_dir_d = f'{out_dir}_{digit}'
        print(f'Generating Training Set for digit {digit}')
        gen_dataset(data_dir=data_dir, out_dir=out_dir_d, digit_class=digit, train=True, mode='id')
        print(f'Generating ID Test Set for digit {digit}')
        gen_dataset(data_dir=data_dir, out_dir=out_dir_d, digit_class=digit, train=False, mode='id')
        print(f'Generating OOD Test Set for digit {digit}')
        gen_dataset(data_dir=data_dir, out_dir=out_dir_d, digit_class=digit, train=False, mode='ood')
        return digit

    with ProcessPoolExecutor(max_workers=min(len(digit_classes), multiprocessing.cpu_count())) as executor:
        futures = [executor.submit(process_digit, digit) for digit in digit_classes]
        for future in as_completed(futures):
            digit = future.result()
            print(f'Finished processing digit {digit}')
