#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this second part only uses the CPU (and also a lot of RAM).  it depends on the
# output of 2-volumes-flow-mesh.py

# usage : ./2-volumes-invmap.py <patch-size> <stride>

import os
import sys
import time

import jax

from connectomics.common import bounding_box
from sofima import map_utils

import importlib

data_loader, patch_size, stride = sys.argv[1:]
patch_size = int(patch_size)
stride = int(stride)

print("data_loader =", data_loader)
print("patch_size =", patch_size)
print("stride =", stride)

data = importlib.import_module(os.path.basename(data_loader))

params = '.patch'+str(patch_size)+'.stride'+str(stride)

flow, mesh = data.load_flow_mesh(params)

box1x = bounding_box.BoundingBox(start=(0, 0, 0, 0),
                                 size=(flow.shape[-1], flow.shape[-2], flow.shape[-3], 1))

# Image warping requires an inverse coordinate map
# does NOT use GPU, but does use a lot of RAM
t0 = time.time()
invmap = map_utils.invert_map(mesh, box1x, box1x, stride)
print("invert_map took", time.time() - t0, "sec")

data.save_invmap(invmap, params)
