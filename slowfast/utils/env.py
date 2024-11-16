#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Set up Environment."""

from iopath.common.file_io import PathManagerFactory

pathmgr = PathManagerFactory.get(key="pyslowfast")
checkpoint_pathmgr = PathManagerFactory.get(key="pyslowfast_checkpoint")
def setup_dist_environment():
    # Add your own env setting.
    pass
