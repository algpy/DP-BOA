import os
import sys
import torchvision
import numpy as np
from copy import deepcopy

from data.data_utils import subsample_balanced_instances, subsample_instances
from config import domainnet_dataroot, domainnet_cut_dataroot

import torch

def _group_indices_by_class(samples):
    class_to_indices = {}
    for idx, (_, cls) in enumerate(samples):
        class_to_indices.setdefault(cls, []).append(idx)
    return class_to_indices

def _balance_domain_indices(major_by_class, minor_by_class, drop_n, rng):
    if drop_n <= 0:
        return major_by_class

    droppable = []
    for cls, major_list in major_by_class.items():
        surplus = len(major_list) - len(minor_by_class.get(cls, []))
        if surplus > 0:
            if surplus >= len(major_list):
                droppable.extend(major_list)
            else:
                droppable.extend(rng.choice(major_list, size=surplus, replace=False).tolist())

    to_drop = []
    if drop_n <= len(droppable):
        to_drop = rng.choice(droppable, size=drop_n, replace=False).tolist()
    else:
        to_drop = list(droppable)
        remaining_drop = drop_n - len(droppable)
        if remaining_drop > 0:
            to_drop_set = set(to_drop)
            remaining = [idx for lst in major_by_class.values() for idx in lst if idx not in to_drop_set]
            if len(remaining) > 0:
                extra = rng.choice(remaining, size=min(remaining_drop, len(remaining)), replace=False).tolist()
                to_drop.extend(extra)

    if to_drop:
        to_drop_set = set(to_drop)
        for cls in major_by_class:
            major_by_class[cls] = [idx for idx in major_by_class[cls] if idx not in to_drop_set]

    return major_by_class

def _select_domainnet_indices_by_class(src_samples, tgt_samples, samples_per_class, seed=0, balance_domains=True):
    rng = np.random.RandomState(seed)
    src_by_class = _group_indices_by_class(src_samples)
    tgt_by_class = _group_indices_by_class(tgt_samples)
    classes = sorted(set(src_by_class.keys()) | set(tgt_by_class.keys()))

    src_selected_by_class = {}
    tgt_selected_by_class = {}
    half = samples_per_class // 2

    for cls in classes:
        src_idxs = src_by_class.get(cls, [])
        tgt_idxs = tgt_by_class.get(cls, [])
        n_src = len(src_idxs)
        n_tgt = len(tgt_idxs)
        n_total = n_src + n_tgt

        if n_total <= samples_per_class:
            src_selected_by_class[cls] = list(src_idxs)
            tgt_selected_by_class[cls] = list(tgt_idxs)
            continue

        if n_src >= half and n_tgt >= half:
            src_selected = rng.choice(src_idxs, size=half, replace=False).tolist()
            tgt_selected = rng.choice(tgt_idxs, size=half, replace=False).tolist()
        else:
            if n_src < half:
                take_src = n_src
                take_tgt = min(samples_per_class - take_src, n_tgt)
            else:
                take_tgt = n_tgt
                take_src = min(samples_per_class - take_tgt, n_src)

            src_selected = rng.choice(src_idxs, size=take_src, replace=False).tolist() if take_src > 0 else []
            tgt_selected = rng.choice(tgt_idxs, size=take_tgt, replace=False).tolist() if take_tgt > 0 else []

        src_selected_by_class[cls] = src_selected
        tgt_selected_by_class[cls] = tgt_selected

    if balance_domains:
        total_src = sum(len(v) for v in src_selected_by_class.values())
        total_tgt = sum(len(v) for v in tgt_selected_by_class.values())
        if total_src > total_tgt:
            drop_n = total_src - total_tgt
            src_selected_by_class = _balance_domain_indices(src_selected_by_class, tgt_selected_by_class, drop_n, rng)
        elif total_tgt > total_src:
            drop_n = total_tgt - total_src
            tgt_selected_by_class = _balance_domain_indices(tgt_selected_by_class, src_selected_by_class, drop_n, rng)

    src_keep = [idx for cls in classes for idx in src_selected_by_class.get(cls, [])]
    tgt_keep = [idx for cls in classes for idx in tgt_selected_by_class.get(cls, [])]
    return src_keep, tgt_keep

def _select_domainnet_samples(src_samples, tgt_samples, samples_per_class, seed=0, balance_domains=True):
    src_keep, tgt_keep = _select_domainnet_indices_by_class(
        src_samples,
        tgt_samples,
        samples_per_class,
        seed=seed,
        balance_domains=balance_domains,
    )
    rng = np.random.RandomState(seed)
    rng.shuffle(src_keep)
    rng.shuffle(tgt_keep)
    src_selected = [src_samples[i] for i in src_keep]
    tgt_selected = [tgt_samples[i] for i in tgt_keep]
    return src_selected, tgt_selected

def _apply_domainnet_sampling(src_dataset, tgt_dataset, samples_per_class, seed=0, balance_domains=True):
    src_selected, tgt_selected = _select_domainnet_samples(
        src_dataset.samples,
        tgt_dataset.samples,
        samples_per_class,
        seed=seed,
        balance_domains=balance_domains,
    )

    src_dataset.samples = src_selected
    src_dataset.targets = [label for _, label in src_selected]
    src_dataset.uq_idxs = np.arange(len(src_selected))

    tgt_dataset.samples = tgt_selected
    tgt_dataset.targets = [label for _, label in tgt_selected]
    tgt_dataset.uq_idxs = np.arange(len(tgt_selected))

    return src_dataset, tgt_dataset

def _apply_single_domain_sampling(dataset, samples_per_class, seed=0):
    if samples_per_class is None or samples_per_class <= 0:
        return dataset

    rng = np.random.RandomState(seed)
    class_to_indices = _group_indices_by_class(dataset.samples)
    selected = []
    for _, idxs in class_to_indices.items():
        if len(idxs) <= samples_per_class:
            selected.extend(idxs)
        else:
            selected.extend(rng.choice(idxs, size=samples_per_class, replace=False).tolist())

    rng.shuffle(selected)
    dataset.samples = [dataset.samples[i] for i in selected]
    dataset.targets = [label for _, label in dataset.samples]
    dataset.uq_idxs = np.arange(len(dataset.samples))
    return dataset

class DomainNetDataset(torchvision.datasets.ImageFolder):

    def __init__(self, *args, domain_label=0, **kwargs):

        # Process metadata json for training images into a DataFrame
        super().__init__(*args, **kwargs)

        self.domain_label = int(domain_label)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):

        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]

        return img, label, self.domain_label, uq_idx

class DomainConcatDataset(torch.utils.data.Dataset):
    def __init__(self, root_src, root_aux, transform=None, sample_per_class=None, sample_seed=0, balance_domains=True):
        """
        root_src: Source-domain directory (args.src_env)
        root_aux: Auxiliary-domain directory (args.aux_env)
        transform: torchvision transform
        """
        super().__init__()
        self.transform = transform

        # Read sample paths from ImageFolder.
        src_dataset = torchvision.datasets.ImageFolder(root=root_src)
        aux_dataset = torchvision.datasets.ImageFolder(root=root_aux)

        src_samples = src_dataset.samples
        aux_samples = aux_dataset.samples
        if sample_per_class is not None and sample_per_class > 0:
            src_samples, aux_samples = _select_domainnet_samples(
                src_samples,
                aux_samples,
                sample_per_class,
                seed=sample_seed,
                balance_domains=balance_domains,
            )

        # Pair each sample path with its domain label.
        self.samples = (
            [(path, 0) for path, _ in src_samples] +
            [(path, 1) for path, _ in aux_samples]
        )
        self.targets = [s[1] for s in self.samples]
        self.uq_idxs = np.arange(len(self.samples))

    def __getitem__(self, idx):
        path, domain_label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        uq_idx = self.uq_idxs[idx]
        return img, domain_label, uq_idx

    def __len__(self):
        return len(self.samples)

class ClassConcatDataset(torch.utils.data.Dataset):
    def __init__(self, root_src, root_aux, transform=None, sample_per_class=None, sample_seed=0, balance_domains=True):
        """
        root_src: Source-domain directory (args.src_env)
        root_aux: Auxiliary-domain directory (args.aux_env)
        transform: torchvision transform
        """
        super().__init__()
        self.transform = transform

        # Read sample paths and class labels from ImageFolder.
        src_dataset = torchvision.datasets.ImageFolder(root=root_src)
        aux_dataset = torchvision.datasets.ImageFolder(root=root_aux)

        src_samples = src_dataset.samples
        aux_samples = aux_dataset.samples
        if sample_per_class is not None and sample_per_class > 0:
            src_samples, aux_samples = _select_domainnet_samples(
                src_samples,
                aux_samples,
                sample_per_class,
                seed=sample_seed,
                balance_domains=balance_domains,
            )

        # Pair paths with class labels, without domain labels.
        self.samples = (
            [(path, cls) for path, cls in src_samples] +
            [(path, cls) for path, cls in aux_samples]
        )
        self.targets = [s[1] for s in self.samples]
        self.uq_idxs = np.arange(len(self.samples))

    def __getitem__(self, idx):
        path, class_label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        uq_idx = self.uq_idxs[idx]
        return img, class_label, uq_idx

    def __len__(self):
        return len(self.samples)
    
class ClassDomainConcatDataset(torch.utils.data.Dataset):
    def __init__(self, root_src, root_aux, transform=None, sample_per_class=None, sample_seed=0, balance_domains=True):
        """
        root_src: Source-domain directory (args.src_env)
        root_aux: Auxiliary-domain directory (args.aux_env)
        transform: torchvision transform
        """
        super().__init__()
        self.transform = transform

        # Load ImageFolder samples and class labels.
        src_dataset = torchvision.datasets.ImageFolder(root=root_src)
        aux_dataset = torchvision.datasets.ImageFolder(root=root_aux)

        src_samples = src_dataset.samples
        aux_samples = aux_dataset.samples
        if sample_per_class is not None and sample_per_class > 0:
            src_samples, aux_samples = _select_domainnet_samples(
                src_samples,
                aux_samples,
                sample_per_class,
                seed=sample_seed,
                balance_domains=balance_domains,
            )

        # Sample format: (path, class_label, domain_label)
        self.samples = (
            [(path, cls, 0) for path, cls in src_samples] +
            [(path, cls, 1) for path, cls in aux_samples]
        )

        self.class_targets = [s[1] for s in self.samples]
        self.domain_targets = [s[2] for s in self.samples]
        self.uq_idxs = np.arange(len(self.samples))

    def __getitem__(self, idx):
        path, class_label, domain_label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        uq_idx = self.uq_idxs[idx]
        return img, class_label, domain_label, uq_idx

    def __len__(self):
        return len(self.samples)

class ClassDomainKNConcatDataset(torch.utils.data.Dataset):
    def __init__(self, root_src, root_aux, train_classes, transform=None, sample_per_class=None, sample_seed=0, balance_domains=True):
        """
        root_src: Source-domain directory (args.src_env)
        root_aux: Auxiliary-domain directory (args.aux_env)
        transform: torchvision transform
        """
        super().__init__()
        self.transform = transform
        self.train_classes = set(train_classes)

        # Read sample paths and class labels from ImageFolder.
        src_dataset = torchvision.datasets.ImageFolder(root=root_src)
        aux_dataset = torchvision.datasets.ImageFolder(root=root_aux)

        src_samples = src_dataset.samples
        aux_samples = aux_dataset.samples
        if sample_per_class is not None and sample_per_class > 0:
            src_samples, aux_samples = _select_domainnet_samples(
                src_samples,
                aux_samples,
                sample_per_class,
                seed=sample_seed,
                balance_domains=balance_domains,
            )

        self.samples = []

        # Source domain.
        for path, cls in src_samples:
            known_novel = 0 if cls in self.train_classes else 1
            domain = 0
            self.samples.append((path, cls, known_novel, domain))

        # Auxiliary domain.
        for path, cls in aux_samples:
            known_novel = 0 if cls in self.train_classes else 1
            domain = 1
            self.samples.append((path, cls, known_novel, domain))
        
        self.targets = [s[1] for s in self.samples]
        self.uq_idxs = np.arange(len(self.samples))

    def __getitem__(self, idx):
        path, class_label, known_novel, domain = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        uq_idx = self.uq_idxs[idx]
        return img, class_label, known_novel, domain, uq_idx

    def __len__(self):
        return len(self.samples)

class DualTransformDataset(torch.utils.data.Dataset):
    """
    Expose two views of the same dataset with separate transforms.
    e.g. DomainConcatDataset(train_transform) + DomainConcatDataset(another_transform)
    """
    def __init__(self, base_dataset, transform1, transform2=None):
        super().__init__()
        self.base_dataset = deepcopy(base_dataset)
        self.transform1 = transform1
        self.transform2 = transform2

        # Remove the base transform to avoid applying it twice.
        self.base_dataset.transform = None

    def __len__(self):
        # Expose both transformed views.
        return 2 * len(self.base_dataset)

    def __getitem__(self, idx):
        # Use transform1 for the first half and transform2 for the second.
        base_idx = idx % len(self.base_dataset)
        sample = self.base_dataset[base_idx]
        if len(sample) == 4:
            img, label, domain_label, uq_idx = sample
        elif len(sample) == 3:
            img, label, uq_idx = sample
            domain_label = None
        else:
            raise ValueError(f"Unexpected sample size: {len(sample)}")

        transform = self.transform1 if idx < len(self.base_dataset) else self.transform2
        assert transform is not None, f"idx < len(self.base_dataset) = '{idx < len(self.base_dataset)}'"
        img = transform(img)
        
        if domain_label is None:
            return img, label, uq_idx
        return img, label, domain_label, uq_idx

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset

def subsample_dataset_by_uqidx(dataset, uq_idx_list):
    """
    Select a dataset subset by unique sample indices.
    :param dataset: Dataset with a uq_idxs attribute.
    :param uq_idx_list: Unique indices to retain (list or NumPy array).
    :return: Selected dataset.
    """
    uq_idx_set = set(uq_idx_list)

    # Find the matching positions.
    mask = np.array([u in uq_idx_set for u in dataset.uq_idxs])

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()
    dataset.uq_idxs = np.array(dataset.uq_idxs)[mask]

    # Restore integer label types.
    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset

def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5, val_ratio=None):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        n_val = max(1, int(len(cls_idxs) * val_ratio)) if val_ratio is not None else val_instances_per_class
        # Have a balanced test set
        v_ = np.random.choice(cls_idxs, replace=False, size=n_val)
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def subsample_dataset_domainclass(dataset, idxs):
    # Use a mask to preserve the original sample order.
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    # Filter samples: (path, class_label, domain_label).
    # Keep tuples intact so paths do not coerce integer labels to strings.
    dataset.samples = [s for i, s in enumerate(dataset.samples) if mask[i]]

    # Filter class labels.
    dataset.class_targets = [t for i, t in enumerate(dataset.class_targets) if mask[i]]
    
    # Filter domain labels.
    dataset.domain_targets = [d for i, d in enumerate(dataset.domain_targets) if mask[i]]

    # Apply the same mask to unique indices.
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset

def get_train_val_indices_domainclass(train_dataset, val_instances_per_class=5, val_ratio=None):
    

    train_classes = list(set(train_dataset.class_targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        

        # Convert the label list to an array for elementwise comparison.
        cls_idxs = np.where(np.array(train_dataset.class_targets) == cls)[0]

        n_val = max(1, int(len(cls_idxs) * val_ratio)) if val_ratio is not None else val_instances_per_class
        
        # Limit validation samples to the available class size.
        if n_val > len(cls_idxs):
            n_val = len(cls_idxs)

        # Have a balanced test set
        v_ = np.random.choice(cls_idxs, replace=False, size=n_val)
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_domainnet_datasets(train_transform, test_transform, train_classes=range(345//2), prop_train_labels=0.8, seed=0, split_train_val=False, args=None):

    np.random.seed(seed)
    samples_per_class = getattr(args, 'domainnet_samples_per_class', 100)
    balance_domains = getattr(args, 'domainnet_balance_domains', True)
    sample_seed = getattr(args, 'domainnet_sample_seed', seed)
    known_src_labelled_per_class = getattr(args, 'domainnet_known_src_labelled_per_class', None)
    sampling_enabled = samples_per_class is not None and samples_per_class > 0

    # Get test dataset
    if args.only_test:
        # Get test dataset
        test_datasetC = DomainNetDataset(transform=test_transform, root=os.path.join(domainnet_dataroot, args.tgt_env), domain_label=1)
        all_datasets = {
            'test': test_datasetC,
        }
        return all_datasets

    if args.task_type == 'A_L+A_U->B':
        train_dataset = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.src_env), domain_label=0)

        if sampling_enabled:
            train_dataset = _apply_single_domain_sampling(train_dataset, samples_per_class, seed=sample_seed)
        if args.use_partial_dataset and not sampling_enabled:
            # subsample the whole training set to accelerate training
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset), prop_indices_to_subsample=0.1)
            train_dataset = subsample_dataset(train_dataset, subsample_whole_indices)

        # Get labelled training set which has subsampled classes, then subsample some indices from that
        train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
        subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

        # Split into training and validation sets
        if split_train_val:
            train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled, val_instances_per_class=5)
            train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
            val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
            val_dataset_labelled_split.transform = test_transform

        else:
            train_dataset_labelled_split, val_dataset_labelled_split = None, None

        # Get unlabelled data 
        unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))

        # Transform dict
        # unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
        # target_xform_dict = {}
        # for i, k in enumerate(list(train_classes) + unlabelled_classes):
        #     target_xform_dict[k] = i

        # train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]

        # Either split train into train and val or use test set as val
        train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

        # Get test dataset
        test_datasetB = DomainNetDataset(transform=test_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'trainA_unlabelled': train_dataset_unlabelled,
            'val': val_dataset_labelled,
            'testB': test_datasetB,
        }

    elif args.task_type == 'A_L+A_U+B->A_U+B+C':
        train_dataset = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.src_env), domain_label=0)
        train_dataset2 = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

        if sampling_enabled:
            train_dataset, train_dataset2 = _apply_domainnet_sampling(
                train_dataset,
                train_dataset2,
                samples_per_class,
                seed=sample_seed,
                balance_domains=balance_domains,
            )
        if args.use_partial_dataset and not sampling_enabled:
            # subsample the whole training set to accelerate training
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset), prop_indices_to_subsample=0.1)
            sub_train_dataset = subsample_dataset(deepcopy(train_dataset), subsample_whole_indices)
        else:
            sub_train_dataset = train_dataset

        # Get labelled training set which has subsampled classes, then subsample some indices from that
        train_dataset_labelled = subsample_classes(deepcopy(sub_train_dataset), include_classes=train_classes)
        if known_src_labelled_per_class is not None:
            subsample_indices = subsample_instances(train_dataset_labelled, sample_per_class=int(known_src_labelled_per_class))
        else:
            subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

        # Get labelled training set which has subsampled classes, then subsample some indices from that
        if sampling_enabled:
            train_dataset_selected = train_dataset2
        else:
            subsample_indices2 = subsample_balanced_instances(deepcopy(train_dataset2), prop_indices_to_subsample=0.3)
            train_dataset_selected = subsample_dataset(train_dataset2, subsample_indices2)

        # Get unlabelled data 
        unlabelled_indices = set(sub_train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        train_datasetA_unlabelled = subsample_dataset(deepcopy(sub_train_dataset), np.array(list(unlabelled_indices)))
        train_datasetB_unlabelled = train_dataset_selected

        # Get test dataset
        test_datasetB  = DomainNetDataset(transform=test_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

        # Transform dict
        # unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
        # target_xform_dict = {}
        # for i, k in enumerate(list(train_classes) + unlabelled_classes):
        #     target_xform_dict[k] = i

        # train_datasetA_unlabelled.target_transform = lambda x: target_xform_dict[x]
        # train_datasetB_unlabelled.target_transform = lambda x: target_xform_dict[x]

        # test_datasetB.target_transform = lambda x: target_xform_dict[x]
        
        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'trainA_unlabelled': train_datasetA_unlabelled,
            'trainB_unlabelled': train_datasetB_unlabelled,
            'testB': test_datasetB,
            'test_seen': train_datasetA_unlabelled,
        }
    
    
    elif args.task_type == 'A_L+A_U+B->A_U+B+C(cut)':
        train_dataset = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.src_env), domain_label=0)
        train_dataset2 = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

        if sampling_enabled:
            train_dataset, train_dataset2 = _apply_domainnet_sampling(
                train_dataset,
                train_dataset2,
                samples_per_class,
                seed=sample_seed,
                balance_domains=balance_domains,
            )
        if args.use_partial_dataset and not sampling_enabled:
            # subsample the whole training set to accelerate training
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset), prop_indices_to_subsample=0.1)
            sub_train_dataset = subsample_dataset(deepcopy(train_dataset), subsample_whole_indices)
        else:
            sub_train_dataset = train_dataset

        # Get labelled training set which has subsampled classes, then subsample some indices from that
        train_dataset_labelled = subsample_classes(deepcopy(sub_train_dataset), include_classes=train_classes)
        if known_src_labelled_per_class is not None:
            subsample_indices = subsample_instances(train_dataset_labelled, sample_per_class=int(known_src_labelled_per_class))
        else:
            subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
        
        # -------------------------------------------------------------------------
        # LOAD TRANSFORMED (CUT) DATA
        # -------------------------------------------------------------------------
        train_labelled_transformed = deepcopy(train_dataset_labelled)
        # Using domainnet_cut_dataroot defined in config.py
        cut_root = os.path.join(domainnet_cut_dataroot, args.aux_env, 'fake_B')

        if not os.path.exists(cut_root):
            raise FileNotFoundError(f"CUT data root not found: {cut_root}")

        new_samples = []
        warn_count = 0
        
        # Iterate over original labelled samples and check if transformed version exists
        for path, label in train_dataset_labelled.samples:
            file_name = os.path.basename(path) # e.g. train_001.jpg
            parent_folder = os.path.basename(os.path.dirname(path)) # e.g. aircraft
            
            name_no_ext = os.path.splitext(file_name)[0]
            # Constructed unique name without extension
            unique_base = f"{parent_folder}_{name_no_ext}" 
            # Final expected filename in fake_B (usually .png)
            expected_filename = unique_base + '.png'
            
            cut_path = os.path.join(cut_root, expected_filename)
            
            if os.path.exists(cut_path):
                new_samples.append((cut_path, label))
            else:
                if warn_count < 5:
                    print(f"Warning: Transformed image not found for {path} (expected at {cut_path}). Skipping.")
                warn_count += 1
        
        if warn_count >= 5:
            print(f"Warning: ... and {warn_count - 5} more similar warnings.")

        print(f"Loaded {len(new_samples)} transformed samples from {cut_root} (matched with {len(train_dataset_labelled)} labelled samples).")
        
        train_labelled_transformed.samples = new_samples
        train_labelled_transformed.targets = [label for _, label in new_samples]
        train_labelled_transformed.uq_idxs = np.array(range(len(new_samples)))
        # -------------------------------------------------------------------------

        # Get labelled training set which has subsampled classes, then subsample some indices from that
        if sampling_enabled:
            train_dataset_selected = train_dataset2
        else:
            subsample_indices2 = subsample_balanced_instances(deepcopy(train_dataset2), prop_indices_to_subsample=0.3)
            train_dataset_selected = subsample_dataset(train_dataset2, subsample_indices2)

        # Get unlabelled data 
        unlabelled_indices = set(sub_train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        train_datasetA_unlabelled = subsample_dataset(deepcopy(sub_train_dataset), np.array(list(unlabelled_indices)))
        train_datasetB_unlabelled = train_dataset_selected

        # Get test dataset
        test_datasetB  = DomainNetDataset(transform=test_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

        # Transform dict
        # unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
        # target_xform_dict = {}
        # for i, k in enumerate(list(train_classes) + unlabelled_classes):
        #     target_xform_dict[k] = i

        # train_datasetA_unlabelled.target_transform = lambda x: target_xform_dict[x]
        # train_datasetB_unlabelled.target_transform = lambda x: target_xform_dict[x]

        # test_datasetB.target_transform = lambda x: target_xform_dict[x]
        
        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'train_labelled_transformed': train_labelled_transformed,
            'trainA_unlabelled': train_datasetA_unlabelled,
            'trainB_unlabelled': train_datasetB_unlabelled,
            'testB': test_datasetB,
            'test_seen': train_datasetA_unlabelled,
        }
    
    elif args.task_type == 'A_L+B_L+A_U+B_U->A_U+B_U':
        if args.pretrain:
            train_dataset = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.src_env), domain_label=0)
            train_dataset2 = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

            if sampling_enabled:
                train_dataset, train_dataset2 = _apply_domainnet_sampling(
                    train_dataset,
                    train_dataset2,
                    samples_per_class,
                    seed=sample_seed,
                    balance_domains=balance_domains,
                )
            # Get labelled training set which has subsampled classes, then subsample some indices from that
            train_dataset_labelled_total = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
            if known_src_labelled_per_class is not None:
                subsample_indices = subsample_instances(train_dataset_labelled_total, sample_per_class=int(known_src_labelled_per_class))
            else:
                subsample_indices = subsample_instances(train_dataset_labelled_total, prop_indices_to_subsample=prop_train_labels)
            train_dataset_labelled = subsample_dataset(deepcopy(train_dataset_labelled_total), subsample_indices)

            # Get labelled training set which has subsampled classes, then subsample some indices from that
            train_dataset2_labelled_total = subsample_classes(deepcopy(train_dataset2), include_classes=train_classes)
            subsample_indices2 = subsample_instances(train_dataset2_labelled_total, prop_indices_to_subsample=prop_train_labels)
            train_dataset2_labelled = subsample_dataset(deepcopy(train_dataset2_labelled_total), subsample_indices2)

            # Get unlabelled data            
            val_indices = set(train_dataset_labelled_total.uq_idxs) - set(train_dataset_labelled.uq_idxs)
            train_datasetA_val = subsample_dataset_by_uqidx(deepcopy(train_dataset_labelled_total), np.array(list(val_indices)))
            
            val_indices = set(train_dataset2_labelled_total.uq_idxs) - set(train_dataset2_labelled.uq_idxs)
            train_datasetB_val = subsample_dataset_by_uqidx(deepcopy(train_dataset2_labelled_total), np.array(list(val_indices)))

            train_datasetA_val.transform = test_transform
            train_datasetB_val.transform = test_transform
            
            # Get test dataset
            test_datasetB  = DomainNetDataset(transform=test_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)
            
            all_datasets = {
                'trainA_labelled_train': train_dataset_labelled,
                'trainB_labelled_train': train_dataset2_labelled,
                'trainA_labelled_val': train_datasetA_val,
                'trainB_labelled_val': train_datasetB_val,
            }

        else:
            train_dataset = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.src_env), domain_label=0)
            train_dataset2 = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

            if sampling_enabled:
                train_dataset, train_dataset2 = _apply_domainnet_sampling(
                    train_dataset,
                    train_dataset2,
                    samples_per_class,
                    seed=sample_seed,
                    balance_domains=balance_domains,
                )
            if args.use_partial_dataset and not sampling_enabled:
                # subsample the whole training set to accelerate training
                subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset), prop_indices_to_subsample=0.2)
                train_dataset = subsample_dataset(deepcopy(train_dataset), subsample_whole_indices)
                subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset2), prop_indices_to_subsample=0.2)
                train_dataset2 = subsample_dataset(deepcopy(train_dataset2), subsample_whole_indices)

            # Get labelled training set which has subsampled classes, then subsample some indices from that
            train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
            if known_src_labelled_per_class is not None:
                subsample_indices = subsample_instances(train_dataset_labelled, sample_per_class=int(known_src_labelled_per_class))
            else:
                subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
            train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

            # Get labelled training set which has subsampled classes, then subsample some indices from that
            train_dataset2_labelled = subsample_classes(deepcopy(train_dataset2), include_classes=train_classes)
            subsample_indices2 = subsample_instances(train_dataset2_labelled, prop_indices_to_subsample=prop_train_labels)
            train_dataset2_labelled = subsample_dataset(train_dataset2_labelled, subsample_indices2)

            # Get unlabelled data 
            unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
            train_datasetA_unlabelled = subsample_dataset_by_uqidx(deepcopy(train_dataset), np.array(list(unlabelled_indices)))
            
            unlabelled_indices = set(train_dataset2.uq_idxs) - set(train_dataset2_labelled.uq_idxs)
            train_dataset2_unlabelled = subsample_dataset_by_uqidx(deepcopy(train_dataset2), np.array(list(unlabelled_indices)))

            train_datasetB_unlabelled = train_dataset2
            # Get test dataset
            test_datasetB  = DomainNetDataset(transform=test_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)
            
            all_datasets = {
                'train_labelled': train_dataset_labelled,
                'trainA_unlabelled': train_datasetA_unlabelled,
                'trainB_unlabelled': train_datasetB_unlabelled,
                'trainB_seen_labelled': train_dataset2_labelled,
                'trainB_unlabelled_modified': train_dataset2_unlabelled,
                'testB': test_datasetB,
                'test_seen': train_datasetA_unlabelled,
            }
    
    elif args.task_type == 'linear_probe':
        train_dataset = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.src_env), domain_label=0)
        train_dataset2 = DomainNetDataset(transform=train_transform, root=os.path.join(domainnet_dataroot, args.aux_env), domain_label=1)

        if sampling_enabled:
            train_dataset, train_dataset2 = _apply_domainnet_sampling(
                train_dataset,
                train_dataset2,
                samples_per_class,
                seed=sample_seed,
                balance_domains=balance_domains,
            )
        if args.use_partial_dataset and not sampling_enabled:
            # subsample the whole training set to accelerate training
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset), prop_indices_to_subsample=0.2)
            train_dataset = subsample_dataset(deepcopy(train_dataset), subsample_whole_indices)
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset2), prop_indices_to_subsample=0.2)
            train_dataset2 = subsample_dataset(deepcopy(train_dataset2), subsample_whole_indices)

        train_datasetA_seen = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)

        train_datasetB_seen = subsample_classes(deepcopy(train_dataset2), include_classes=train_classes)

        # Get unlabelled data 
        unlabelled_indices = set(train_dataset.uq_idxs) - set(train_datasetA_seen.uq_idxs)
        train_datasetA_unseen = subsample_dataset_by_uqidx(deepcopy(train_dataset), np.array(list(unlabelled_indices)))
        
        unlabelled_indices = set(train_dataset2.uq_idxs) - set(train_datasetB_seen.uq_idxs)
        train_datasetB_unseen = subsample_dataset_by_uqidx(deepcopy(train_dataset2), np.array(list(unlabelled_indices)))
        
        train_idxs, val_idxs = get_train_val_indices(train_datasetA_seen, val_ratio=0.2)
        train_datasetA_seen_train = subsample_dataset(deepcopy(train_datasetA_seen), train_idxs)
        train_datasetA_seen_val = subsample_dataset(deepcopy(train_datasetA_seen), val_idxs)
        train_datasetA_seen_val.transform = test_transform
        
        train_idxs, val_idxs = get_train_val_indices(train_datasetB_seen, val_ratio=0.2)
        train_datasetB_seen_train = subsample_dataset(deepcopy(train_datasetB_seen), train_idxs)
        train_datasetB_seen_val = subsample_dataset(deepcopy(train_datasetB_seen), val_idxs)
        train_datasetB_seen_val.transform = test_transform
        
        train_idxs, val_idxs = get_train_val_indices(train_datasetA_unseen, val_ratio=0.2)
        train_datasetA_unseen_train = subsample_dataset(deepcopy(train_datasetA_unseen), train_idxs)
        train_datasetA_unseen_val = subsample_dataset(deepcopy(train_datasetA_unseen), val_idxs)
        train_datasetA_unseen_val.transform = test_transform
        
        train_idxs, val_idxs = get_train_val_indices(train_datasetB_unseen, val_ratio=0.2)
        train_datasetB_unseen_train = subsample_dataset(deepcopy(train_datasetB_unseen), train_idxs)
        train_datasetB_unseen_val = subsample_dataset(deepcopy(train_datasetB_unseen), val_idxs)
        train_datasetB_unseen_val.transform = test_transform

        train_dataset_domain = DomainConcatDataset(
            root_src=os.path.join(domainnet_dataroot, args.src_env),
            root_aux=os.path.join(domainnet_dataroot, args.aux_env),
            transform=train_transform,
            sample_per_class=samples_per_class if sampling_enabled else None,
            sample_seed=sample_seed,
            balance_domains=balance_domains
        )

        if args.use_partial_dataset and not sampling_enabled:
            # subsample the whole training set to accelerate training
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset_domain), prop_indices_to_subsample=0.2)
            train_dataset_domain = subsample_dataset(deepcopy(train_dataset_domain), subsample_whole_indices)
        
        train_idxs, val_idxs = get_train_val_indices(train_dataset_domain, val_ratio=0.2)
        train_dataset_domain_train = subsample_dataset(deepcopy(train_dataset_domain), train_idxs)
        train_dataset_domain_val = subsample_dataset(deepcopy(train_dataset_domain), val_idxs)
        train_dataset_domain_val.transform = test_transform
        
        all_datasets = {
            'train_datasetA_seen_train': train_datasetA_seen_train,
            'train_datasetA_seen_val': train_datasetA_seen_val,
            'train_datasetB_seen_train': train_datasetB_seen_train,
            'train_datasetB_seen_val': train_datasetB_seen_val,
            'train_datasetA_unseen_train': train_datasetA_unseen_train,
            'train_datasetA_unseen_val': train_datasetA_unseen_val,
            'train_datasetB_unseen_train': train_datasetB_unseen_train,
            'train_datasetB_unseen_val': train_datasetB_unseen_val,
            'train_dataset_domain_train': train_dataset_domain_train,
            'train_dataset_domain_val': train_dataset_domain_val,
        }
    
    elif args.task_type == 'domain_class':
        # Domain
        train_dataset_domain = DomainConcatDataset(
            root_src=os.path.join(domainnet_dataroot, args.src_env),
            root_aux=os.path.join(domainnet_dataroot, args.aux_env),
            transform=train_transform,
            sample_per_class=samples_per_class if sampling_enabled else None,
            sample_seed=sample_seed,
            balance_domains=balance_domains
        )

        if args.use_partial_dataset and not sampling_enabled:
            # subsample the whole training set to accelerate training
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset_domain), prop_indices_to_subsample=0.2)
            train_dataset_domain = subsample_dataset(deepcopy(train_dataset_domain), subsample_whole_indices)
        
        train_idxs, val_idxs = get_train_val_indices(train_dataset_domain, val_ratio=0.2)
        train_dataset_domain_train = subsample_dataset(deepcopy(train_dataset_domain), train_idxs)
        train_dataset_domain_val = subsample_dataset(deepcopy(train_dataset_domain), val_idxs)
        train_dataset_domain_val.transform = test_transform
        
        # Class
        train_dataset_class = ClassConcatDataset(
            root_src=os.path.join(domainnet_dataroot, args.src_env),
            root_aux=os.path.join(domainnet_dataroot, args.aux_env),
            transform=train_transform,
            sample_per_class=samples_per_class if sampling_enabled else None,
            sample_seed=sample_seed,
            balance_domains=balance_domains
        )

        train_idxs, val_idxs = get_train_val_indices(train_dataset_class, val_ratio=0.2)
        train_dataset_class_train = subsample_dataset(deepcopy(train_dataset_class), train_idxs)
        train_dataset_class_val = subsample_dataset(deepcopy(train_dataset_class), val_idxs)
        train_dataset_class_val.transform = test_transform
        
        all_datasets = {
            'train_dataset_class_train': train_dataset_class_train,
            'train_dataset_class_val': train_dataset_class_val,
            'train_dataset_domain_train': train_dataset_domain_train,
            'train_dataset_domain_val': train_dataset_domain_val,
        }
    
    elif args.task_type == 'domain_class_both':
        # Domain
        train_dataset = ClassDomainConcatDataset(
            root_src=os.path.join(domainnet_dataroot, args.src_env),
            root_aux=os.path.join(domainnet_dataroot, args.aux_env),
            transform=train_transform,
            sample_per_class=samples_per_class if sampling_enabled else None,
            sample_seed=sample_seed,
            balance_domains=balance_domains
        )

        # if args.use_partial_dataset:
        #     # subsample the whole training set to accelerate training
        #     subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset), prop_indices_to_subsample=0.2)
        #     train_dataset = subsample_dataset(deepcopy(train_dataset), subsample_whole_indices)
        
        train_idxs, val_idxs = get_train_val_indices_domainclass(train_dataset, val_ratio=0.2)
        train_dataset_train = subsample_dataset_domainclass(deepcopy(train_dataset), train_idxs)
        train_dataset_val = subsample_dataset_domainclass(deepcopy(train_dataset), val_idxs)
        train_dataset_val.transform = test_transform
        
        all_datasets = {
            'train_dataset_train': train_dataset_train,
            'train_dataset_val': train_dataset_val,
        }
    
    elif args.task_type == 'knst':
        # Class
        train_dataset = ClassDomainKNConcatDataset(
            root_src=os.path.join(domainnet_dataroot, args.src_env),
            root_aux=os.path.join(domainnet_dataroot, args.aux_env),
            train_classes=train_classes,
            transform=train_transform,
            sample_per_class=samples_per_class if sampling_enabled else None,
            sample_seed=sample_seed,
            balance_domains=balance_domains
        )
        
        all_datasets = {
            'train_dataset': train_dataset,
        }
    
    elif args.task_type == 'domain_class_T':
        # Domain
        train_dataset_domain = DomainConcatDataset(
            root_src=os.path.join(domainnet_dataroot, args.src_env),
            root_aux=os.path.join(domainnet_dataroot, args.aux_env),
            transform=None,
            sample_per_class=samples_per_class if sampling_enabled else None,
            sample_seed=sample_seed,
            balance_domains=balance_domains
        )

        if args.use_partial_dataset and not sampling_enabled:
            # subsample the whole training set to accelerate training
            subsample_whole_indices = subsample_balanced_instances(deepcopy(train_dataset_domain), prop_indices_to_subsample=0.2)
            train_dataset_domain = subsample_dataset(deepcopy(train_dataset_domain), subsample_whole_indices)
        
        train_idxs, val_idxs = get_train_val_indices(train_dataset_domain, val_ratio=0.2)
        
        train_dataset_domain_train = subsample_dataset(deepcopy(train_dataset_domain), train_idxs)
        train_dataset_domain_train = DualTransformDataset(train_dataset_domain_train, test_transform, None)
        
        train_dataset_domain_val = subsample_dataset(deepcopy(train_dataset_domain), val_idxs)
        train_dataset_domain_val.transform = test_transform
        
        # Class
        train_dataset_class = ClassConcatDataset(
            root_src=os.path.join(domainnet_dataroot, args.src_env),
            root_aux=os.path.join(domainnet_dataroot, args.aux_env),
            transform=None,
            sample_per_class=samples_per_class if sampling_enabled else None,
            sample_seed=sample_seed,
            balance_domains=balance_domains
        )

        train_idxs, val_idxs = get_train_val_indices(train_dataset_class, val_ratio=0.2)
        
        train_dataset_class_train = subsample_dataset(deepcopy(train_dataset_class), train_idxs)
        train_dataset_class_train = DualTransformDataset(train_dataset_class_train, test_transform, None)
        
        train_dataset_class_val = subsample_dataset(deepcopy(train_dataset_class), val_idxs)
        train_dataset_class_val.transform = test_transform
        
        all_datasets = {
            'train_dataset_class_train': train_dataset_class_train,
            'train_dataset_class_val': train_dataset_class_val,
            'train_dataset_domain_train': train_dataset_domain_train,
            'train_dataset_domain_val': train_dataset_domain_val,
        }
    
    return all_datasets


if __name__ == '__main__':

    from types import SimpleNamespace
    from collections import Counter
    args = SimpleNamespace( 
        only_test=False,
        task_type='linear_probe',
        src_env='real',
        aux_env='clipart',
        use_partial_dataset=False
    )

    datasets = get_domainnet_datasets(None, None, args=args)

    # ============================================
    # Print label distributions.
    # ============================================
    def print_label_distribution(dataset, name):
        counter = Counter(dataset.targets)
        print(f"\n{name} label distribution:")
        for k, v in sorted(counter.items()):
            print(f"  class {k}: {v}")

    print_label_distribution(datasets['train_dataset_domain_train'], 'train_dataset_domain_train')
    print_label_distribution(datasets['train_dataset_domain_val'], 'train_dataset_domain_val')
