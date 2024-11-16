# The source code is borrowed from https://github.com/joslefaure/HIT/blob/master/preprocess_data/ava/keypoints_detection.py
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from tqdm import tqdm
import torch
import numpy as np
import os, json, cv2


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2

# Specify counted frames path
video_path = 'val/keyframes'


# Specify output json path
json_path = 'keypoints.json'

# Keep a dictionary of models
models = {"objects": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
          "persons_and_keypoints": "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"}

# Specify task
task = 'persons_and_keypoints'

# list_of_images = os.listdir(path)
cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(models[task]))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(models[task])

predictor = DefaultPredictor(cfg)

all_outputs = dict()
# for image in tqdm(list_of_images):
all_video = os.listdir(video_path)
for video in tqdm(all_video):
    all_outputs[video] = dict()
    all_frames = os.listdir(os.path.join(video_path, video))
    all_frames.sort()
    for frame in all_frames:
        if frame not in all_outputs[video]:
            all_outputs[video][frame] = dict()
        image_path = os.path.join(video_path, video, frame)
        im = cv2.imread(image_path)
        outputs = predictor(im)
        all_outputs[video][frame]['bbox'] = outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist(),
        all_outputs[video][frame]['keypoints'] = outputs["instances"].pred_keypoints.cpu().numpy().tolist(),


json.dump(all_outputs, open("val_keypoint.json", "w"))

