#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this third part only uses the CPU (and also a lot of RAM).  it depends on the
# output of N-planes-mesh.py

import sys
import os
import argparse
import importlib

from connectomics.common import bounding_box
from sofima import map_utils
from datetime import datetime 

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="inverts the coordinate map of the mesh"
)
parser.add_argument(
    "data_loader",
    help="Data loader module name, e.g., data-test-2-planes"
)
parser.add_argument(
    "basepath",
    help="filepath to stitched planes"
)
parser.add_argument(
    "min_z",
    type=int,
    help="lower bound on the planes to align"
)
parser.add_argument(
    "max_z",
    type=int,
    help="upper bound on the planes to align"
)
parser.add_argument(
    "patch_size",
    type=int,
    help="Side length of (square) patch for processing (in pixels, e.g., 32)",
)
parser.add_argument(
    "stride",
    type=int,
    help="Distance of adjacent patches (in pixels, e.g., 8)"
)
parser.add_argument(
    "scales",
    help="the spatial resolutions to use when computing the flow field"
)
parser.add_argument(
    "k0",
    type=float,
    help="spring constant for inter-section springs"
)
parser.add_argument(
    "k",
    type=float,
    help="spring constant for intra-section springs"
)
parser.add_argument(
    "reps",
    type=int,
    help="how many times to iteratively compute the flow"
)

args = parser.parse_args()

data_loader = args.data_loader
basepath = args.basepath
min_z = args.min_z
max_z = args.max_z
patch_size = args.patch_size
stride = args.stride
scales_int = [int(x) for x in args.scales.split(',')]
k0 = args.k0
k = args.k
reps = args.reps

print("data_loader =", data_loader)
print("basepath =", basepath)
print("min_z =", min_z)
print("max_z =", max_z)
print("patch_size =", patch_size)
print("stride =", stride)
print("scales =", scales_int)
print("k0 =", k0)
print("k =", k)
print("reps =", reps)

data = importlib.import_module(os.path.basename(data_loader))

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+args.scales.replace(",",'')+'.k0'+str(k0)+'.k'+str(k)+'.reps'+str(reps)

flow = data.load_flow(basepath, params)
mesh = data.load_mesh(basepath, params)

boxMx = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

s_min = min(scales_int)
stride_min = stride * (2**s_min)

print(datetime.now(), 'inverting map')
invmap = map_utils.invert_map(mesh, boxMx, boxMx, stride_min)

print(datetime.now(), 'saving inverted map')
data.save_invmap(invmap, basepath, params)
