#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""

from fvcore.common.config import CfgNode

def add_custom_config(_C):
    # Add your own customized configs.
    _C.COMPUTER = CfgNode()
    _C.COMPUTER.ENABLE = False
    _C.COMPUTER.FEATURE_EXTRACTION_TRAIN = 'path/to/human and context features'
    _C.COMPUTER.H5_TEST_PATH = 'path/to/human and context features'
    _C.COMPUTER.WINDOW_SIZE = 3
    _C.COMPUTER.SPARSE_ATTENTION = False
    _C.COMPUTER.LONG_TERM_RANGE = 5
    _C.COMPUTER.CLIP_SELECTION = 'path/to/pre-computed clip selection'
    _C.COMPUTER.TRANS_HEAD_MODE = 'human_context_current'
    _C.COMPUTER.BOX_RELATION = False
    _C.COMPUTER.NUM_RELATION_MODULE = 1
    _C.COMPUTER.OBJECT = "path/to/ava_object.csv"
    _C.COMPUTER.OBJECT_SCORE_THRESH = 0.9
    _C.COMPUTER.SKELETON_VAL = "path the skeleton val set"
    _C.COMPUTER.SKELETON_TRAIN = "path the skeleton train set"


