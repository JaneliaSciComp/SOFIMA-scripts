#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this second part only uses the CPU (and also a lot of RAM).  it depends on the
# output of em-alignment1.py

# usage : ./em-alignment2.py <patch-size> <stride>

import sys
from concurrent import futures
import time

import jax
import jax.numpy as jnp
import numpy as np

from connectomics.common import bounding_box
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils

import data

patch_size, stride = sys.argv[1:]
patch_size = int(patch_size)
stride = int(stride)

print("patch_size =", patch_size)
print("stride =", stride)

params = '.patch'+str(patch_size)+'.stride'+str(stride)

flow, mesh = data.load_flow_mesh(params)

mesh = np.array(mesh.read().result())

box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

# Image warping requires an inverse coordinate map
# does NOT use GPU, but does use a lot of RAM
t0 = time.time()
inv_map = map_utils.invert_map(mesh, box1x, box1x, stride)
print("invert_map took", time.time() - t0, "sec")

data.save_map(inv_map, params)
