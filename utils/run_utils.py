import os
import torch

import tasks
from torch import distributed
from utils.logger import Logger
import numpy as np
import random
from PIL import Image

from torch.utils.data import ConcatDataset
from utils import get_dataset_class, get_augmentation

def load_ckpt(checkpoint, model, strict=True):
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=strict)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    else:
        model.module.load_state_dict(checkpoint, strict=strict)

def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score, prototypes, count_features):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict(),
        "prototypes": prototypes,
        "count_features": count_features
    }
    torch.save(state, path)

def get_dataset(opts, rank):
    """ Dataset And Augmentation
    """
    train_transform, target_transform, val_transform, test_transform = get_augmentation(opts.crop_val)

    labels, labels_old = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels
    
    source_path_base = f'data/{opts.dataset}/{opts.task}'
    source_dataset = get_dataset_class(opts.dataset)

    if opts.uda_target:
        target_path_base = f'data/{opts.target_dataset}/{opts.task}-{opts.dataset}'
        target_dataset = get_dataset_class(opts.target_dataset)
    else:
        target_path_base = f'data/{opts.dataset}/{opts.task}'
        target_dataset = source_dataset

    if opts.overlap:
        source_path_base += "-ov"
        target_path_base += "-ov"

    if opts.no_mask:
        source_path_base += "-oldclICCVW2019"
        target_path_base += "-oldclICCVW2019"

    if not os.path.exists(source_path_base):
        os.makedirs(source_path_base, exist_ok=True)

    if not os.path.exists(target_path_base):
        os.makedirs(target_path_base, exist_ok=True)
        
    if opts.weight_samples > 0:
        source_training_base = source_path_base + "-weighted_"+str(opts.weight_samples)
        if not os.path.exists(source_training_base):
            os.makedirs(source_training_base, exist_ok=True)
        source_dst = source_dataset(root=opts.data_root, train=True, transform=train_transform,
                            labels=list(labels), labels_old=list(labels_old),
                            idxs_path= source_training_base +  f"/train-{opts.step}.npz",
                            masking=not opts.no_mask, overlap=opts.overlap, weight=opts.weight_samples, where_to_sim=opts.where_to_sim, rank=rank, 
                            label_filter = opts.filter_unused_labels)
    else:
        source_dst = source_dataset(root=opts.data_root, train=True, transform=train_transform,
                            labels=list(labels), labels_old=list(labels_old),
                            idxs_path= source_path_base +  f"/train-{opts.step}.npy",
                            masking=not opts.no_mask, overlap=opts.overlap, where_to_sim=opts.where_to_sim, rank=rank, 
                            label_filter = opts.filter_unused_labels)
    if opts.uda_target:
        target_dst = target_dataset(root=opts.data_root, train=True, transform=target_transform,
                                    labels=list(labels), labels_old=list(labels_old),
                                    idxs_path=target_path_base + f"/train-{opts.step}.npy",
                                    masking=not opts.no_mask, overlap=opts.overlap, where_to_sim=opts.where_to_sim, rank=rank, 
                                    label_filter = opts.filter_unused_labels)
    else:
        target_dst = None

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(source_dst))
        val_len = len(source_dst)-train_len
        source_dst, val_dst = torch.utils.data.random_split(source_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = source_dataset(root=opts.data_root, train=False, transform=val_transform,
                           labels=list(labels_cum),
                           idxs_path=source_path_base + f"/val-{opts.step}.npy", label_filter = opts.filter_unused_labels)
        """
        not cumulative val
        val_dst = source_dataset(root=opts.data_root, train=False, transform=val_transform,
                          labels=list(labels), labels_old=list(labels_old),
                          idxs_path=source_path_base + f"/val-{opts.step}.npy",
                          masking=not opts.no_mask, overlap=True, label_filter = opts.filter_unused_labels)
        """
    image_set = 'train' if opts.val_on_trainset else 'val'
    
    if opts.uda_validate_target:
        test_dst = target_dataset(root=opts.data_root, train=opts.val_on_trainset, transform=test_transform,
                           labels=list(labels_cum),
                           idxs_path=target_path_base + f"/test_on_{image_set}-{opts.step}.npy", label_filter = opts.filter_unused_labels)
    else:
        test_dst = source_dataset(root=opts.data_root, train=opts.val_on_trainset, transform=test_transform,
                           labels=list(labels_cum),
                           idxs_path=source_path_base + f"/test_on_{image_set}-{opts.step}.npy", label_filter = opts.filter_unused_labels)        

    return source_dst, target_dst, val_dst, test_dst, len(labels_cum)


def define_distrib_training(opts, logdir_full):
    if opts.where_to_sim == 'GPU_server':
        # here use model with apex and inplace_abn
        distributed.init_process_group(backend='nccl', init_method='env://')
        device_id, device = opts.local_rank, torch.device(opts.local_rank)
        torch.cuda.set_device(device_id)
        #print(torch.cuda.memory_summary(device))
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
    elif opts.where_to_sim == 'GPU_windows':
        device_id, device = opts.local_rank, torch.device(opts.local_rank)
        torch.cuda.set_device(device_id)
        rank, world_size = 0, 1
    elif opts.where_to_sim == 'CPU':
        distributed.init_process_group(backend='gloo', init_method='env://')  # was nccl here
        device = torch.device('cpu')
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
    elif opts.where_to_sim == 'CPU_windows':
        device_id, device = opts.local_rank, torch.device('cpu')
        torch.cuda.set_device(device_id)
        rank, world_size = 0, 1

    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    return device, rank, world_size, logger

def setup_random_seeds(opts):
    # Set up random seed
    torch.manual_seed(opts.random_seed)
    if not opts.where_to_sim == 'CPU':
        torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)