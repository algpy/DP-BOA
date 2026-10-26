from data.data_utils import MergedDataset, StripDomainLabelDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets, get_imagenet_1k_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.inat21 import get_inat21_datasets, inat_root, anno_path, val_annots
from data.food101 import get_food_101_datasets
from data.pets import get_oxford_pets_datasets
from data.inaturalist import get_inaturalist_datasets
from data.domainnet import get_domainnet_datasets

from copy import deepcopy
import pickle
import os
from torch.utils.data import ConcatDataset

from config import osr_split_dir


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'imagenet_1k': get_imagenet_1k_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'iNat21': get_inat21_datasets,
    'Mollusca': get_inaturalist_datasets,
    'Fungi': get_inaturalist_datasets,
    'Arachnida': get_inaturalist_datasets,
    'Animalia': get_inaturalist_datasets,
    'pets': get_oxford_pets_datasets,
    'food': get_food_101_datasets,
    'domainnet': get_domainnet_datasets,
}

def _ensure_domainnet_args(args):
    if not hasattr(args, 'task_type'):
        args.task_type = 'A_L+A_U+B->A_U+B+C'
    if not hasattr(args, 'src_env'):
        args.src_env = 'real'
    if not hasattr(args, 'aux_env'):
        args.aux_env = 'clipart'
    if not hasattr(args, 'tgt_env'):
        args.tgt_env = args.aux_env
    if not hasattr(args, 'only_test'):
        args.only_test = False
    if not hasattr(args, 'use_partial_dataset'):
        args.use_partial_dataset = False
    if not hasattr(args, 'domainnet_samples_per_class'):
        args.domainnet_samples_per_class = 100
    if not hasattr(args, 'domainnet_balance_domains'):
        args.domainnet_balance_domains = True
    if not hasattr(args, 'domainnet_sample_seed'):
        args.domainnet_sample_seed = getattr(args, 'seed', 0)
    if not hasattr(args, 'domainnet_known_src_labelled_per_class'):
        args.domainnet_known_src_labelled_per_class = 40

def _set_dataset_transform(dataset, transform):
    if dataset is None:
        return
    if isinstance(dataset, StripDomainLabelDataset):
        _set_dataset_transform(dataset.dataset, transform)
        return
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            _set_dataset_transform(ds, transform)
        return
    if hasattr(dataset, 'transform'):
        dataset.transform = transform

def _maybe_set_target_transform(dataset, target_transform):
    if dataset is None:
        return
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            _maybe_set_target_transform(ds, target_transform)
        return
    if hasattr(dataset, 'target_transform'):
        dataset.target_transform = target_transform
    else:
        dataset.target_transform = target_transform

def _build_domainnet_outputs(datasets, test_transform, args):
    train_labelled = datasets.get('train_labelled')
    train_unlabelled = datasets.get('train_unlabelled')
    if train_unlabelled is None:
        if 'trainA_unlabelled' in datasets and 'trainB_unlabelled' in datasets:
            train_unlabelled = ConcatDataset([datasets['trainA_unlabelled'], datasets['trainB_unlabelled']])
        elif 'trainA_unlabelled' in datasets:
            train_unlabelled = datasets['trainA_unlabelled']

    test_dataset = datasets.get('test')
    if test_dataset is None:
        test_dataset = datasets.get('testB')

    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for ds in [train_labelled, train_unlabelled, test_dataset]:
        _maybe_set_target_transform(ds, target_transform)

    train_dataset = None
    unlabelled_train_examples_test = None
    if train_labelled is not None and train_unlabelled is not None:
        train_labelled = StripDomainLabelDataset(train_labelled)
        train_unlabelled = StripDomainLabelDataset(train_unlabelled)
        if getattr(args, 'pretrain', False) or getattr(args, 'OCD', False):
            train_dataset = deepcopy(train_labelled)
        else:
            train_dataset = MergedDataset(labelled_dataset=deepcopy(train_labelled),
                                          unlabelled_dataset=deepcopy(train_unlabelled))

        unlabelled_train_examples_test = deepcopy(train_unlabelled)
        _set_dataset_transform(unlabelled_train_examples_test, test_transform)

    if test_dataset is not None:
        test_dataset = StripDomainLabelDataset(test_dataset)

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    if dataset_name == 'domainnet':
        _ensure_domainnet_args(args)
        datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                                 train_classes=args.train_classes,
                                 prop_train_labels=args.prop_train_labels,
                                 split_train_val=False, seed=getattr(args, 'seed', 0), args=args)
        return _build_domainnet_outputs(datasets, test_transform, args)
    if dataset_name == 'cifar100':
        allowed_classes = getattr(args, 'cifar100_allowed_classes', None)
        datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                                train_classes=args.train_classes,
                                prop_train_labels=args.prop_train_labels,
                                split_train_val=False, seed=getattr(args, 'seed', 0),
                                allowed_classes=allowed_classes)
    elif dataset_name in ['Fungi', 'Arachnida', 'Mollusca', 'Animalia']:
        datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                                train_classes=args.train_classes,
                                prop_train_labels=args.prop_train_labels,
                                split_train_val=False, subclassname=dataset_name)
    else:
        datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                                train_classes=args.train_classes,
                                prop_train_labels=args.prop_train_labels,
                                split_train_val=False)
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    try:
        if args.pretrain or args.OCD:
            train_dataset = deepcopy(datasets['train_labelled'])
        else:
            train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                            unlabelled_dataset=deepcopy(datasets['train_unlabelled']))
    except AttributeError:
        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                    unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform
    args.logger.info("Lens: train_labelled: {}, train_unlabelled: {}, test: {}".format(len(datasets['train_labelled']), len(datasets['train_unlabelled']), len(test_dataset)))
    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        if getattr(args, 'cifar100_60_20_split', False):
            args.train_classes = range(70)
            args.unlabeled_classes = range(70, 80)
            args.cifar100_allowed_classes = range(80)
        else:
            args.train_classes = range(args.num_labeled_classes)
            args.unlabeled_classes = range(args.num_labeled_classes, 100)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'imagenet_1k':

        args.image_size = 224
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)

    elif args.dataset_name == 'iNat21':
        args.image_size = 224

        # split_path = os.path.join(osr_split_dir, 'inat_centriod_split.pkl')
        split_path = os.path.join(osr_split_dir, 'inat_splits.pkl')
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        key = args.split_mode  # 'easy', medium, hard or fine-grain
        if key in class_info:
            args.train_classes = class_info[key]['train']
            args.unlabeled_classes = class_info[key]['unlabel']
        else:
            print(f"No key '{key}' in pre define splits")

    
    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)
    elif args.dataset_name == 'food':
        
        args.train_classes = range(51)
        args.unlabeled_classes = range(51, 101)

    elif args.dataset_name == 'pets':

        args.train_classes = range(19)
        args.unlabeled_classes = range(19, 37)

    elif args.dataset_name == 'Animalia':

        args.train_classes = range(39)
        args.unlabeled_classes = range(39, 77)

    elif args.dataset_name == 'Arachnida':

        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    elif args.dataset_name == 'Fungi':
        
        args.train_classes = range(61)
        args.unlabeled_classes = range(61, 121)

    elif args.dataset_name == 'Mollusca':

        args.train_classes = range(47)
        args.unlabeled_classes = range(47, 93)

    elif args.dataset_name == 'domainnet':

        args.image_size = 224
        args.train_classes = range(345 // 2)
        args.unlabeled_classes = range(345 // 2, 345)

    else:

        raise NotImplementedError

    return args
