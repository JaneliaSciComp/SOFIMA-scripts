#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this third part only uses the CPU (and also a lot of RAM).  it depends on the
# output of N-planes-mesh.py

# usage : ./N-planes-invmap.py <data-loader> <basepath> <min-z> <max-z> <patch-size> <stride> <scales> <k0> <k> <reps>

import sys
import os
import importlib

from connectomics.common import bounding_box
from sofima import map_utils
from datetime import datetime 

data_loader, basepath, min_z, max_z, patch_size, stride, scales_str, k0, k, reps = sys.argv[1:]
patch_size = int(patch_size)
stride = int(stride)
scales = [int(x) for x in scales_str.split(',')]

print("data_loader =", data_loader)
print("basepath =", basepath)
print("min_z =", min_z)
print("max_z =", max_z)
print("patch_size =", patch_size)
print("stride =", stride)
print("scales =", scales_str)
print("k0 =", k0)
print("k =", k)

data = importlib.import_module(os.path.basename(data_loader))

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+str(scales_str).replace(",",'')+'.k0'+str(k0)+'.k'+str(k)+'.reps'+reps

flow = data.load_flow(basepath, params)
mesh = data.load_mesh(basepath, params)

boxMx = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

s_min = min(scales)
stride_min = stride * (2**s_min)

print(datetime.now(), 'inverting map')
invmap = map_utils.invert_map(mesh, boxMx, boxMx, stride_min)

print(datetime.now(), 'saving inverted map')
data.save_invmap(invmap, basepath, params)
