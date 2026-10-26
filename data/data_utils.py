import numpy as np
from torch.utils.data import Dataset

def _extract_targets(dataset):
    if hasattr(dataset, 'targets'):
        return dataset.targets
    if hasattr(dataset, 'target'):
        return dataset.target
    if hasattr(dataset, 'samples'):
        return [s[1] for s in dataset.samples]
    if hasattr(dataset, 'data'):
        try:
            import pandas as pd
            if isinstance(dataset.data, pd.DataFrame) and 'target' in dataset.data:
                return dataset.data['target'].values
        except Exception:
            pass
    targets = []
    for i in range(len(dataset)):
        try:
            _, target, *_ = dataset[i]
            targets.append(target)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot extract a label from dataset[{i}]; expected (data, label, ...).")
    return targets

def subsample_instances(dataset, prop_indices_to_subsample=0.8, sample_per_class=None):
    """
    Subsample a dataset.

    Args:
        dataset: Dataset whose labels are obtained from supported attributes.
        prop_indices_to_subsample (float): Sampling fraction when sample_per_class is None.
        sample_per_class (int): Number of samples per class. When set, overrides
                                 prop_indices_to_subsample.

    Returns:
        np.ndarray: Indices of the selected samples.
    """
    # Use a fixed seed for reproducible sampling.
    np.random.seed(0)

    if sample_per_class is None:
        # Sample the requested fraction when no per-class count is specified.
        num_samples_to_subsample = int(prop_indices_to_subsample * len(dataset))
        subsample_indices = np.random.choice(
            range(len(dataset)), 
            replace=False,
            size=num_samples_to_subsample
        )
    else:

        # Sample a fixed number of instances per class.
        
        # Read labels from the dataset attributes.
        targets = None
        if hasattr(dataset, 'targets'):
            # Standard torchvision datasets, such as CIFAR10.
            print("Reading labels from dataset.targets.")
            targets = dataset.targets
        elif hasattr(dataset, 'target'):

            # CarsDataset stores labels in target.
            print("Reading labels from dataset.target.")
            targets = dataset.target
        elif hasattr(dataset, 'samples'):
            # FGVCAircraft and ImageFolder store (path, label) pairs.
            print("Reading labels from dataset.samples.")
            targets = [s[1] for s in dataset.samples]
        elif hasattr(dataset, 'data') and isinstance(dataset.data, pd.DataFrame) and 'target' in dataset.data:
            # CustomCub2011 stores labels in a DataFrame.
            print("Reading labels from dataset.data['target'].")
            targets = dataset.data['target'].values
        else:
            # Fall back to reading labels through the dataset interface.
            print("Warning: no standard label attribute found; iterating over the dataset may be slow.")
            targets = []
            for i in range(len(dataset)):
                # Expect __getitem__ to return (data, label, ...).
                try:
                    _, target, *_ = dataset[i]
                    targets.append(target)
                except (ValueError, TypeError):
                    raise ValueError(f"Cannot extract a label from dataset[{i}]; expected (data, label, ...).")

        if targets is None:
            raise ValueError("Cannot extract labels from the dataset.")
            
        targets = np.array(targets)
        unique_classes = np.unique(targets)
        
        subsample_indices = []
        # Sample each class independently.
        for cls in unique_classes:
            # Find the indices of this class.
            class_indices = np.where(targets == cls)[0]
            
            # Keep all available samples if the class is smaller than requested.
            if len(class_indices) < sample_per_class:
                print(f"Warning: class {cls} has only {len(class_indices)} samples, "
                      f"fewer than the requested {sample_per_class}. Selecting all available samples.")
                num_to_sample = len(class_indices)
            else:
                num_to_sample = sample_per_class

            # Choose indices from this class.
            # Sample without replacement.
            chosen_indices = np.random.choice(
                class_indices, 
                size=num_to_sample, 
                replace=False
            )
            
            subsample_indices.extend(chosen_indices)
        
        # Shuffle the selected indices to mix classes.
        subsample_indices = np.array(subsample_indices)
        np.random.shuffle(subsample_indices)

    return subsample_indices

def subsample_balanced_instances(dataset, prop_indices_to_subsample=0.8, seed=0):
    if prop_indices_to_subsample <= 0:
        return np.array([], dtype=np.int64)

    rng = np.random.RandomState(seed)
    targets = np.array(_extract_targets(dataset))
    unique_classes = np.unique(targets)

    subsample_indices = []
    for cls in unique_classes:
        cls_idxs = np.where(targets == cls)[0]
        if len(cls_idxs) == 0:
            continue
        num_to_sample = int(prop_indices_to_subsample * len(cls_idxs))
        if prop_indices_to_subsample > 0 and num_to_sample == 0:
            num_to_sample = 1
        num_to_sample = min(len(cls_idxs), num_to_sample)
        chosen_indices = rng.choice(cls_idxs, size=num_to_sample, replace=False)
        subsample_indices.extend(chosen_indices.tolist())

    subsample_indices = np.array(subsample_indices, dtype=np.int64)
    rng.shuffle(subsample_indices)
    return subsample_indices

class StripDomainLabelDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.uq_idxs = getattr(base_dataset, 'uq_idxs', np.arange(len(base_dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if not isinstance(sample, (list, tuple)) or len(sample) < 3:
            raise ValueError(f"Unexpected sample format: {sample}")
        if len(sample) == 3:
            return sample
        img = sample[0]
        label = sample[1]
        uq_idx = sample[-1]
        return img, label, uq_idx

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
