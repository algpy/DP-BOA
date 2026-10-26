import os

data_root = os.environ.get('DATA_ROOT', './datasets')
pretrained_root = os.environ.get('PRETRAINED_ROOT', './pretrained')

# Dataset locations can be overridden independently without editing source code.
cifar_10_root = os.environ.get('CIFAR10_ROOT', os.path.join(data_root, 'cifar10'))
cifar_100_root = os.environ.get('CIFAR100_ROOT', os.path.join(data_root, 'cifar100'))
cub_root = os.environ.get('CUB_ROOT', os.path.join(data_root, 'cub'))
aircraft_root = os.environ.get('AIRCRAFT_ROOT', os.path.join(data_root, 'fgvc-aircraft-2013b'))
herbarium_dataroot = os.environ.get('HERBARIUM_ROOT', os.path.join(data_root, 'herbarium_19'))
imagenet_root = os.environ.get('IMAGENET_ROOT', os.path.join(data_root, 'ILSVRC2012'))
tinyimagenet_root = os.environ.get('TINYIMAGENET_ROOT', os.path.join(data_root, 'tiny-imagenet-200'))
car_root = os.environ.get('CARS_ROOT', os.path.join(data_root, 'stanford_cars'))
oxfordiiitpet_root = os.environ.get('PETS_ROOT', os.path.join(data_root, 'Oxford-IIIT-Pet'))
inaturalist_root = os.environ.get('INATURALIST_ROOT', os.path.join(data_root, 'iNat17'))
food101_root = os.environ.get('FOOD101_ROOT', os.path.join(data_root, 'food-101'))
domainnet_dataroot = os.environ.get('DOMAINNET_ROOT', os.path.join(data_root, 'domain_net'))
domainnet_cut_dataroot = os.environ.get('DOMAINNET_CUT_ROOT', os.path.join(data_root, 'domainnet_oracle'))

# The iNat21 loader concatenates this root with relative image filenames.
inat21_root = os.path.join(os.environ.get('INAT21_ROOT', os.path.join(data_root, 'iNature')), '')
inat21_train_annotations = os.environ.get('INAT21_TRAIN_ANNOTATIONS', os.path.join(inat21_root, 'train_mini.json'))
inat21_val_annotations = os.environ.get('INAT21_VAL_ANNOTATIONS', os.path.join(inat21_root, 'val.json'))

osr_split_dir = os.environ.get('OSR_SPLIT_DIR', 'data/ssb_splits')
dino_pretrain_path = os.environ.get('DINO_PRETRAIN_PATH', os.path.join(pretrained_root, 'dino_vitbase16_pretrain.pth'))
dinov2_vitb14_pretrain_path = os.environ.get('DINOV2_PRETRAIN_PATH', os.path.join(pretrained_root, 'dinov2_vitb14_pretrain.pth'))

exp_root = os.environ.get('EXP_ROOT', './checkpoints')          # All logs and checkpoints will be saved here

ovr_envs = ["real", "painting", "quickdraw", "sketch", "clipart", "infograph"] 
