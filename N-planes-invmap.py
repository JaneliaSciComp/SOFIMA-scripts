#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this third part only uses the CPU (and also a lot of RAM).  it depends on the
# output of N-planes-mesh.py

# usage : ./N-planes-invmap.py <data-loader> <min-z> <max-z> <patch-size> <stride>

import os
import importlib

from connectomics.common import bounding_box
from sofima import map_utils
from datetime import datetime 

data_loader, min_z, max_z, patch_size, stride = sys.argv[1:]
patch_size = int(patch_size)
stride = int(stride)

print("data_loader =", data_loader)
print("min_z =", min_z)
print("max_z =", max_z)
print("patch_size =", patch_size)
print("stride =", stride)

data = importlib.import_module(os.path.basename(data_loader))

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)

flow = data.load_flow(params)
mesh = data.load_mesh(params)

box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1)) # f1

print(datetime.now(), 'inverting map')
invmap = map_utils.invert_map(mesh, box1x, box1x, stride)

print(datetime.now(), 'saving inverted map')
data.save_invmap(invmap, params)
