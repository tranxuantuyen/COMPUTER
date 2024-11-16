#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch

from . import ava_helper as ava_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY
import h5py
import torch
logger = logging.getLogger(__name__)
import os
import json
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
def find_overlapping_boxes(target_box, boxes):
    """Find all boxes that overlap with the target box."""
    overlapping_boxes = []
    for box in boxes:
        if box[0] < target_box[2] and box[2] > target_box[0] and box[1] < target_box[3] and box[3] > target_box[1]:
            overlapping_boxes.append(box)
    return overlapping_boxes

def create_containing_box(target_box, overlapping_boxes):
    """Create a new box that contains the target box and all overlapping boxes."""
    containing_box = list(target_box)
    for box in overlapping_boxes:
        containing_box[0] = min(containing_box[0], box[0])
        containing_box[1] = min(containing_box[1], box[1])
        containing_box[2] = max(containing_box[2], box[2])
        containing_box[3] = max(containing_box[3], box[3])
    return containing_box

@DATASET_REGISTRY.register()
class Ava(torch.utils.data.Dataset):
    """
    AVA Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.AVA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.AVA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP

        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        self._image_paths, self._video_idx_to_name = ava_helper.load_image_lists(
            cfg, is_train=(self._split == "train")
        )

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        assert len(boxes_and_labels) == len(self._image_paths)

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== AVA dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            # random flip
            imgs, boxes = cv2_transform.horizontal_flip_list(
                0.5, imgs, order="HWC", boxes=boxes
            )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
        boxes = np.array(boxes)
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()
        if self.cfg.COMPUTER.OBJECT_SCORE_THRESH > 0 and self.cfg.COMPUTER:
            boxes_object = []
            clip_label_list_object = self._keyframe_boxes_and_labels_object[video_idx][sec_idx]
            for box_labels_object in clip_label_list_object:
                boxes_object.append(box_labels_object[0])
            boxes_object = np.array(boxes_object)
            # Score is not used.
            boxesObject = boxes_object[:, :4].copy()

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND
        )
        # if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
        #     # T H W C -> T C H W.
        #     # Preprocess images and boxes.
        #     _, boxesObject = self._images_and_boxes_preprocessing(
        #         imgs, boxes=boxesObject
        #     )
        #     # T C H W -> C T H W.
        # else:
        #     # Preprocess images and boxes
        #     _, boxesObject = self._images_and_boxes_preprocessing_cv2(
        #         imgs, boxes=boxesObject
        #     )

        if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(
                imgs, boxes=boxes
            )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                imgs, boxes=boxes
            )

        # Construct label arrays.
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
                label_arrs[i][label - 1] = 1

        imgs = utils.pack_pathway_output(self.cfg, imgs)
        metadata = [[video_idx, sec]] * len(boxes)

        extra_data = {
            "boxes": boxes,
            # "boxesObject": boxesObject,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
        }

        return imgs, label_arrs, idx, extra_data
    
def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

@DATASET_REGISTRY.register()
class Avawithfeature(Ava):
    def __init__(self, cfg, split) -> None:
        super().__init__(cfg, split)
        
        if cfg.COMPUTER.SPARSE_ATTENTION:
            filename = 'train.h5' if split == 'train' else 'test.h5'
            self.context_one_feature = load_dict_from_hdf5(os.path.join(cfg.COMPUTER.CONTEXT_ONE_FEATURE, filename))
        if split == 'train':
            self.h5_data_path = cfg.COMPUTER.H5_TRAIN_PATH 
            if cfg.COMPUTER.SKELETON:
                with open(cfg.COMPUTER.SKELETON_TRAIN) as f:
                    self.skeleton = json.load(f)
            # self.skeleton = json.load(open(cfg.COMPUTER.SKELETON_TRAIN))
        else:    
            self.h5_data_path = cfg.COMPUTER.H5_TEST_PATH 
            if cfg.COMPUTER.SKELETON:
                with open(cfg.COMPUTER.SKELETON_VAL) as f:
                    self.skeleton = json.load(f)
        self.window_size = cfg.COMPUTER.WINDOW_SIZE

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        self._image_paths, self._video_idx_to_name = ava_helper.load_image_lists(
            cfg, is_train=(self._split == "train")
        )

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        if cfg.COMPUTER.OBJECT_SCORE_THRESH > 0 and cfg.COMPUTER:
            boxes_and_labels_object = ava_helper.load_boxes_and_labels(
                cfg, mode=self._split
            )
            boxes_object = ava_helper.load_object_boxes(
                cfg, mode=self._split
            )

            # boxes_and_labels_temp = [i[0] for i in boxes_and_labels]
            # boxes_object_temp = [i[0] for i in boxes_and_labels]
            for video_name in boxes_and_labels.keys():
                for frame_sec in boxes_and_labels[video_name].keys():
                    # Save in format of a list of [box_i, box_i_labels].
                    boxes_and_labels_temp = [i[0] for i in boxes_and_labels[video_name][frame_sec]]
                    boxes_object_temp = [i[0] for i in boxes_object[video_name][frame_sec]]
                    overlapping_boxes = []
                    for target_box in boxes_and_labels_temp:
                        overlapping_boxes.append(find_overlapping_boxes(target_box, boxes_object_temp))

                    # Create a new box that contains each target box and all overlapping boxes.
                    containing_boxes = []
                    for i, target_box in enumerate(boxes_and_labels_temp):
                        containing_boxes.append(create_containing_box(target_box, overlapping_boxes[i]))
                    for i in range(len(boxes_and_labels[video_name][frame_sec])):
                        # boxes_and_labels[video_name][frame_sec][i][0] = containing_boxes[i]
                        boxes_and_labels_object[video_name][frame_sec][i][0] = containing_boxes[i]
                        ##############################################
                
            boxes_and_labels_object = [
            boxes_and_labels_object[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
            ]

            (
            _,
            self._keyframe_boxes_and_labels_object,
            )   = ava_helper.get_keyframe_data(boxes_and_labels_object)


        assert len(boxes_and_labels) == len(self._image_paths)

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()
    def __getitem__(self, idx):
        imgs, label_arrs, idx, extra_data = super().__getitem__(idx)
        time = 0
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        path_to_file = os.path.join(self.h5_data_path, str(video_idx), str(sec)+ '.h5')
        h5feature = load_dict_from_hdf5(path_to_file)
        human_feature = h5feature[str(video_idx)][str(sec)]['human_feature']
        context_feature = h5feature[str(video_idx)][str(sec)]['context'].squeeze()

        if self.cfg.COMPUTER.SPARSE_ATTENTION:
            past_context_one_vector = []
            future_context_one_vector = []
            current_context_one_vector = self.context_one_feature[str(video_idx)][str(sec)]['context_one_feature']
            for i in range(sec - self.cfg.COMPUTER.LONG_TERM_RANGE, sec):
                if str(i) in self.context_one_feature[str(video_idx)].keys():
                    past_context_one_vector.append(self.context_one_feature[str(video_idx)][str(i)]['context_one_feature'])
            if past_context_one_vector:
                past_context_one = np.vstack(past_context_one_vector)      
                past_score = past_context_one @ current_context_one_vector 
                if len(past_score) > self.window_size: 
                    past_score_index = np.argpartition(past_score, -self.window_size)[-self.window_size:]
                    
                else:
                    past_score_index = range(len(past_score))
                past_sec = [range(sec - self.cfg.COMPUTER.LONG_TERM_RANGE, sec)[i] for i in past_score_index]
                
            else:
                past_sec = range(sec - self.window_size, sec)
            for i in range(sec + 1, sec + self.cfg.COMPUTER.LONG_TERM_RANGE + 1):
                if str(i) in self.context_one_feature[str(video_idx)].keys():
                    future_context_one_vector.append(self.context_one_feature[str(video_idx)][str(i)]['context_one_feature'])
            if future_context_one_vector:
                future_context_one = np.vstack(future_context_one_vector)   
                future_score = future_context_one @ current_context_one_vector 
                if len(future_score) > self.window_size:    
                    future_score_index = np.argpartition(future_score, -self.window_size)[-self.window_size:]
                else:
                    future_score_index = range(len(future_score))
                future_sec = [range(sec + 1, sec + self.cfg.COMPUTER.LONG_TERM_RANGE + 1)[i] for i in future_score_index]
                
            else:
                future_sec = range(sec + 1, sec + self.window_size + 1)
        else:
            past_sec = range(sec - self.window_size, sec)
            future_sec = range(sec + 1, sec + self.window_size + 1)

        human_feature_past_list = []
        human_feature_future_list = []
        contex_feature_past_list = []
        contex_feature_future_list = []
        if self.cfg.COMPUTER.BOX_RELATION:
            box_past_list = []
            box_future_list = []
        if self.cfg.COMPUTER.SKELETON:
            past_skeleton_list = []
            future_skeleton_list = []
            try:
                current_skeleton = np.array(self.skeleton[self._video_idx_to_name[video_idx]][str(sec)+ '.jpg']['keypoints'])[0, :, :, :2]
            except:
                current_skeleton = np.zeros((extra_data['ori_boxes'].shape[0], 17, 2))
            if current_skeleton.shape[0] > extra_data['ori_boxes'].shape[0]:
                current_skeleton = current_skeleton[:extra_data['ori_boxes'].shape[0], :, :]
            elif current_skeleton.shape[0] < extra_data['ori_boxes'].shape[0]:
                pad_shape = (extra_data['ori_boxes'].shape[0] - current_skeleton.shape[0], current_skeleton.shape[1], current_skeleton.shape[2])
                pad_value = current_skeleton[-1, :, :]
                current_skeleton = np.concatenate((current_skeleton, np.tile(pad_value, (pad_shape[0], 1, 1))), axis=0)
                # current_skeleton = np.pad(current_skeleton, ((0, pad_shape[0]), (0, 0), (0, 0)), 'constant', constant_values=pad_value)
        for i in past_sec:
            file_path = os.path.join(self.h5_data_path, str(video_idx), str(i)+ '.h5')
            if not os.path.isfile(file_path):
                continue
            feature = load_dict_from_hdf5(file_path)
            human_feature_past_list.append(feature[str(video_idx)][str(i)]['human_feature'])
            contex_feature_past_list.append(feature[str(video_idx)][str(i)]['context'].squeeze())
            if self.cfg.COMPUTER.BOX_RELATION:
                box_past_list.append(feature[str(video_idx)][str(i)]['meta']['ori_boxes'])
            if self.cfg.COMPUTER.SKELETON:
                try:
                    past_skeleton_list.append(np.array(self.skeleton[self._video_idx_to_name[video_idx]][str(i)+ '.jpg']['keypoints'])[0, :, :, :2])
                except:
                    past_skeleton_list = [np.array(current_skeleton)]
        if not human_feature_past_list:
            human_feature_past = torch.tensor(human_feature)
            contex_feature_past = torch.tensor(context_feature)
            if self.cfg.COMPUTER.BOX_RELATION:
                box_past = torch.tensor(extra_data['ori_boxes'])
            if self.cfg.COMPUTER.SKELETON:
                past_skeleton = np.array(current_skeleton)
        else: 
            human_feature_past = torch.tensor(np.concatenate(human_feature_past_list))
            contex_feature_past = torch.tensor(np.concatenate(contex_feature_past_list))
            if self.cfg.COMPUTER.BOX_RELATION:
                box_past = torch.tensor(np.concatenate(box_past_list))[:, 1:]
            if self.cfg.COMPUTER.SKELETON:
                past_skeleton = np.concatenate(past_skeleton_list)
        for i in future_sec:
            file_path = os.path.join(self.h5_data_path, str(video_idx), str(i)+ '.h5')
            if not os.path.isfile(os.path.join(self.h5_data_path, str(video_idx), str(i)+ '.h5')):
                continue
            feature = load_dict_from_hdf5(file_path)
            human_feature_future_list.append(feature[str(video_idx)][str(i)]['human_feature'])
            contex_feature_future_list.append(feature[str(video_idx)][str(i)]['context'].squeeze())
            if self.cfg.COMPUTER.BOX_RELATION:
                box_future_list.append(feature[str(video_idx)][str(i)]['meta']['ori_boxes'])
            if self.cfg.COMPUTER.SKELETON:
                try:
                    future_skeleton_list.append(np.array(self.skeleton[self._video_idx_to_name[video_idx]][str(i)+ '.jpg']['keypoints'])[0, :, :, :2])
                except:
                    future_skeleton_list = [np.array(current_skeleton)]


        if not human_feature_future_list:
            human_feature_future = torch.tensor(human_feature)
            contex_feature_future = torch.tensor(context_feature)
            if self.cfg.COMPUTER.BOX_RELATION:
                box_future = torch.tensor(extra_data['ori_boxes'])
            if self.cfg.COMPUTER.SKELETON:
                future_skeleton = np.array(current_skeleton)
        else: 
            human_feature_future = torch.tensor(np.concatenate(human_feature_future_list))
            contex_feature_future = torch.tensor(np.concatenate(contex_feature_future_list))
            if self.cfg.COMPUTER.BOX_RELATION:
                box_future = torch.tensor(np.concatenate(box_future_list))[:, 1:]
            if self.cfg.COMPUTER.SKELETON:
                future_skeleton = np.concatenate(future_skeleton_list)
        if self.cfg.COMPUTER.BOX_RELATION:
            extra_data['box_future'] = box_future
            extra_data['box_past'] = box_past
        if self.cfg.COMPUTER.SKELETON:
            past_skeleton = past_skeleton - past_skeleton[:, 8, :][:, np.newaxis, :]
            past_skeleton[:, :, 0] = past_skeleton[:, :, 0] / IMAGE_WIDTH   
            past_skeleton[:, :, 1] = past_skeleton[:, :, 1] / IMAGE_HEIGHT
            past_skeleton_box = np.copy(past_skeleton)
            past_skeleton_box[:, :, 0] = past_skeleton[:, :, 0] - extra_data['ori_boxes'][0, 0]
            past_skeleton_box[:, :, 1] = past_skeleton[:, :, 1] - extra_data['ori_boxes'][0, 1]
            past_skeleton = np.concatenate((past_skeleton, past_skeleton_box), axis=1)
            past_skeleton = past_skeleton.reshape(-1, 34*2)

            future_skeleton = future_skeleton - future_skeleton[:, 8, :][:, np.newaxis, :]
            future_skeleton[:, :, 0] = future_skeleton[:, :, 0] / IMAGE_WIDTH
            future_skeleton[:, :, 1] = future_skeleton[:, :, 1] / IMAGE_HEIGHT
            future_skeleton_box = np.copy(future_skeleton)
            future_skeleton_box[:, :, 0] = future_skeleton[:, :, 0] - extra_data['ori_boxes'][0, 0]
            future_skeleton_box[:, :, 1] = future_skeleton[:, :, 1] - extra_data['ori_boxes'][0, 1]
            future_skeleton = np.concatenate((future_skeleton, future_skeleton_box), axis=1)
            future_skeleton = future_skeleton.reshape(-1, 34*2)

            current_skeleton = current_skeleton - current_skeleton[:, 8, :][:, np.newaxis, :]
            current_skeleton[:, :, 0] = current_skeleton[:, :, 0] / IMAGE_WIDTH
            current_skeleton[:, :, 1] = current_skeleton[:, :, 1] / IMAGE_HEIGHT
            current_skeleton_box = np.copy(current_skeleton)
            current_skeleton_box[:, :, 0] = current_skeleton[:, :, 0] - extra_data['ori_boxes'][0, 0]
            current_skeleton_box[:, :, 1] = current_skeleton[:, :, 1] - extra_data['ori_boxes'][0, 1]
            current_skeleton = np.concatenate((current_skeleton, current_skeleton_box), axis=1)
            current_skeleton = current_skeleton.reshape(-1, 34*2)

            extra_data['past_skeleton'] = past_skeleton
            extra_data['future_skeleton'] = future_skeleton
            extra_data['current_skeleton'] = current_skeleton



        inputs = imgs, (human_feature_past, human_feature_future), (contex_feature_past, contex_feature_future)
        return inputs, label_arrs, idx, time, extra_data
