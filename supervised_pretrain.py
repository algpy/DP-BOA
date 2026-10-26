import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import vision_transformer as vits

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from data.data_utils import StripDomainLabelDataset

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.utils import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, ETF_Classifier, model_statistics, TeacherHead
from config import *
import os
import random

def make_deterministic(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def train(student, train_loader, test_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    for epoch in range(args.epochs):
        student.train()
        loss_record = AverageMeter()

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs= batch

            class_labels = class_labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)

                # clustering, sup
                loss = nn.CrossEntropyLoss()(student_out / 0.1, class_labels)

                pstr = ''
                pstr += f'cls_loss: {loss.item():.4f} '
                
            # concept_covariance_matrix = (torch.cov(student_proj.T)) ** 2
            # num_concept = concept_covariance_matrix.shape[0]
            # concept_regularization_loss = (concept_covariance_matrix.sum() - concept_covariance_matrix.diagonal().sum()) / (num_concept * (num_concept - 1))
            
            # avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            # me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            # loss += args.memax_weight * me_max_loss
            #args.logger.info('concept_regularization_loss{}'.format(concept_regularization_loss))
            
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        if (epoch + 1) % 20 == 0:
            args.logger.info('Testing on disjoint test set...')
            acc = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

            args.logger.info('Test Accuracies: {:.4f}'.format(acc))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = np.mean(preds == targets)
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--num_labeled_classes', default=80, type=int)    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--pretrain', action='store_true', default=True)
    parser.add_argument('--split_mode', default='easy', type=str)
    parser.add_argument('--head_dim', default=128, type=int)
    
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--src_env', type=str, default='real')
    parser.add_argument('--aux_env', type=str, default='clipart')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'], exp_id=args.exp_name)
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    
    make_deterministic(args.seed)
    
    pretrain_path = dino_pretrain_path

    backbone = vits.__dict__['vit_base']()

    state_dict = torch.load(pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, generator=g, worker_init_fn=seed_worker, persistent_workers=False
    )
    test_seen_dataset = datasets["test_seen"]
    if args.dataset_name == 'domainnet':
        test_seen_dataset = StripDomainLabelDataset(test_seen_dataset)
    test_loader_labelled = DataLoader(test_seen_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    # projector = TeacherHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, middle_dim=args.head_dim, nlayer=1).to(device)
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes)
    # projector = ETF_Classifier(args.feat_dim, args.mlp_out_dim)
    model = nn.Sequential(backbone, projector).to(device)
    model_statistics(model)
    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, args)
