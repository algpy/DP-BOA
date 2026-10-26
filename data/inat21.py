import os
import numpy as np
from copy import deepcopy

import json
import pickle
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import time

from config import inat21_root as inat_root
from config import inat21_train_annotations as anno_path
from config import inat21_val_annotations as val_annots
chosen_class_path = 'data/ssb_splits/all_inat_classes.pkl'
split_path = 'data/ssb_splits/inat_splits.pkl'

# def default_loader(path):
#     return Image.open(path).convert('RGB')

class CustomiNat(Dataset):
    def __init__(self, data_root, anno_root, is_train=True, transform=None):

        # load annotations
        # print('Loading annotations from: ' + os.path.basename(anno_root))
        with open(anno_root) as data_file:
            annotations = json.load(data_file)

        # Load file name, image id and class id
        self.file_name = [anno['file_name'] for anno in annotations['images']]
        self.ids = [anno['id'] for anno in annotations['images']]
        self.target = [anno['category_id'] for anno in annotations['annotations']]

        # load category info
        self.root = data_root
        self.is_train = is_train
        self.loader = default_loader

        self.uq_idxs = np.array(range(len(self)))

        # augmentation params
        self.transform = transform
        self.target_transform = None

    def __getitem__(self, index):
        # start_time = time.time()
        path = self.root + self.file_name[index]
        idx = self.ids[index]
        image = self.loader(path)
        target = self.target[index]
        # print(image.shape)
        # end_time = time.time()
        # print('Load data cost', end_time - start_time, 's')

        # start_time = time.time()
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # end_time = time.time()
        # print('Tranform data cost', end_time - start_time, 's')

        return image, target, idx

    def __len__(self):
        return len(self.file_name)


def subsample_dataset(dataset, idxs):

    dataset.file_name = np.array(dataset.file_name)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes)
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_inat21_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.5,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomiNat(data_root=inat_root, anno_root= anno_path, transform=train_transform, is_train=True)
    with open(chosen_class_path, 'rb') as f:
        all_classes = pickle.load(f)
    all_dataset = subsample_classes(deepcopy(whole_training_set), include_classes=all_classes)


    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(all_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    # whole_test = CustomiNat(data_root=inat_root, anno_root=anno_path, transform=test_transform, is_train=False)
    # test_dataset_seen = subsample_classes(deepcopy(whole_test), include_classes=train_classes)

    #test dataset: use provided val
    val_data = CustomiNat(data_root=inat_root, anno_root=val_annots, transform=test_transform, is_train=False)
    test_dataset = subsample_classes(deepcopy(val_data), include_classes=all_classes)
    test_dataset_seen = subsample_classes(deepcopy(val_data), include_classes=train_classes)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    # print("Lens: train_labelled: {}, train_unlabelled: {}, val: {}, test: {}, test_seen: {}".format(len(train_dataset_labelled), len(train_dataset_unlabelled), len(val_dataset_labelled) if split_train_val else 0, len(test_dataset), len(test_dataset_seen)))

    print("Lens: train_labelled: {}, train_unlabelled: {}, val: {}".format(
        len(train_dataset_labelled), len(train_dataset_unlabelled), len(val_dataset_labelled) if split_train_val else 0))
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        "test_seen": test_dataset_seen,
    }

    return all_datasets

if __name__ == '__main__':
    with open(split_path, 'rb') as f:
        splits = pickle.load(f)
    train_classes = splits['10']['train']
    x = get_inat21_datasets(None, None, train_classes=train_classes, prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
