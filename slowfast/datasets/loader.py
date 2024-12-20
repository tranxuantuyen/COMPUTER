#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from .build import build_dataset


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx,1, collated_extra_data

def detection_collate_feature(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)

    imgs, human_feature, contex_feature = zip(*inputs)
    past_human, future_human = zip(*human_feature)
    past_contex, future_context = zip(*contex_feature)
    
    past_human, future_human = torch.nn.utils.rnn.pad_sequence(past_human, batch_first=True, padding_value=29.01), torch.nn.utils.rnn.pad_sequence(future_human, batch_first=True, padding_value=29.01)
    past_contex, future_context = torch.nn.utils.rnn.pad_sequence(past_contex, batch_first=True, padding_value=29.01), torch.nn.utils.rnn.pad_sequence(future_context, batch_first=True, padding_value=29.01)

    past_contex_mask = (past_contex == 29.01).all(dim=-1)
    future_context_mask = (future_context == 29.01).all(dim=-1)
    past_humanx_mask = (past_human == 29.01).all(dim=-1)
    future_human_mask = (future_human == 29.01).all(dim=-1)
# =========================
    imgs, video_idx = default_collate(imgs), default_collate(video_idx)
    time = default_collate(time)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        # elif key == 'box_future' or key == 'box_past':
        #     collated_extra_data[key] = torch.nn.utils.rnn.pad_sequence([torch.tensor(i, dtype=torch.float32) for i in data], batch_first=True, padding_value=29.01)

        elif key == "boxes"  or key == "boxesObject":
            data_pad = torch.nn.utils.rnn.pad_sequence([torch.tensor(i, dtype=torch.float32) for i in data], batch_first=True, padding_value=29.01)
            current_human_mask = (data_pad == 29.01).all(dim=-1)
            bboxes = [
                torch.cat(
                    [torch.full((data_pad[i].shape[0], 1), float(i)), data_pad[i]], dim=1
                )
                for i in range(len(data_pad))
            ]
            collated_extra_data[key] = torch.cat(bboxes, dim=0)
            collated_extra_data["current_human_mask"] = current_human_mask
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)
    collated_extra_data["past_contex_mask"] = past_contex_mask
    collated_extra_data["future_context_mask"] = future_context_mask
    collated_extra_data["past_humanx_mask"] = past_humanx_mask
    collated_extra_data["future_human_mask"] = future_human_mask
    final_input = (imgs, past_human, future_human, past_contex, future_context)
    return final_input, labels, video_idx, time, collated_extra_data

def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    if cfg.DETECTION.ENABLE:
        if cfg.COMPUTER.ENABLE:
            collate_func = detection_collate_feature
        else:
            collate_func = detection_collate
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn= collate_func if cfg.DETECTION.ENABLE else None,
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
