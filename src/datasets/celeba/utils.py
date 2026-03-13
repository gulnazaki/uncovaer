import pandas as pd
import random
from .config import SEED


PARTITIONS = {
    '0': 'train',
    '1': 'valid',
    '2': 'test'
}

def load_celebA_attributes(path, split='all', partition_file=None):
    if split != 'all':
        partition_dict = {}
        with open(partition_file, 'r') as f:
            for line in f:
                filename, partition = line.strip().split()
                partition_dict[filename] = PARTITIONS[partition]

    with open(path, 'r') as f:
        lines = f.readlines()

    # Skip the first line (count), use second as header
    header = lines[1].strip().split()
    data = []

    for line in lines[2:]:
        parts = line.strip().split()
        if split != 'all' and partition_dict[parts[0]] != split:
            continue
        else:
            values = [int(x) for x in parts[1:]]  # skip filename
            data.append(values)

    df = pd.DataFrame(data, columns=header)

    df.replace(-1, 0, inplace=True)
    return df

def get_balanced_celeba_indices(df, split, attr_1, attr_2):
    """
    Load CelebA attributes and partition files, and return balanced test indices
    where attr_1 and attr_2 are decorrelated (e.g., Male and Blurry).
    """
    df['index'] = df.index

    # Create combined attribute label
    df['combined'] = df[attr_1].astype(str) + "_" + df[attr_2].astype(str)

    print(f"Balancing {split} set:")
    print(f"{attr_1}_{attr_2}")
    print(df['combined'].value_counts())

    # Balance the test set by subsampling to smallest group size
    min_count = df['combined'].value_counts().min()

    balanced_df = (
        df.groupby('combined')
               .apply(lambda x: x.sample(n=min_count, random_state=SEED))
               .reset_index(drop=True)
    )

    # Return the original indices (relative to full dataset), shuffling them
    indices = balanced_df['index'].tolist()
    rng = random.Random(SEED)
    rng.shuffle(indices)
    return indices

def top_confounders(attr, attr_names, label="Attractive", topk=10):
    # Align lengths
    if len(attr_names) > attr.shape[1]:
        attr_names = attr_names[:attr.shape[1]]
    elif len(attr_names) < attr.shape[1]:
        raise ValueError("attr_names shorter than attr dimension!")

    df = pd.DataFrame(attr.numpy(), columns=attr_names)
    # full correlation matrix
    corr = df.corr()

    # sort attributes by |corr with label|
    sorted_attrs = corr[label].drop(label).abs().sort_values(ascending=False).index

    result = {}
    for attr_name in sorted_attrs:
        # correlations of this attr with all others (except itself)
        others = corr[attr_name].drop(attr_name).abs().sort_values(ascending=False)
        result[attr_name] = list(others.head(topk).index)

    return result