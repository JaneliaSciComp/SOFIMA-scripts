#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this third part only uses the CPU (and also a lot of RAM).  it depends on the
# output of N-planes-invmap.py

import sys
import os
import argparse
import numpy as np
from connectomics.common import bounding_box
from sofima import warp
from datetime import datetime

import importlib

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="warps a slice according to a coordinate map"
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
parser.add_argument(
    "chunk_size",
    type=int,
    help="of the zarr output",
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
chunk_size = args.chunk_size

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
print("chunk_size =", chunk_size)

nz = max_z - min_z + 1
data = importlib.import_module(os.path.basename(data_loader))

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+args.scales.replace(",",'')+'.k0'+str(k0)+'.k'+str(k)+'.reps'+str(reps)
flow = data.load_mesh(basepath, params)
invmap = data.load_invmap(basepath, params)

boxMx = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

print(datetime.now(), 'warping planes')

s_min = min(scales_int)
stride_min = stride * (2**s_min)

warped0 = data.load_data(basepath, min_z,0)
warped = np.zeros((chunk_size, *warped0.shape), dtype=warped0.dtype)
warped[min_z % chunk_size,...] = warped0
fid = data.open_warp([max_z+1, *warped0.shape], chunk_size, basepath, params)

data_box = bounding_box.BoundingBox(start=(0, 0, 0), size=[*np.flip(np.array(warped0.shape)),1])
out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=[*np.flip(np.array(warped0.shape)),1])

for z in range(min_z+1, max_z+1):
  print(datetime.now(), 'z =', z)

  curr = np.transpose(np.expand_dims(np.expand_dims(data.load_data(basepath, z,0),
                                                    axis=-1),
                                     axis=-1),
                      [3, 2, 0, 1])
  warped[z % chunk_size, ...] = warp.warp_subvolume(curr, data_box,
      invmap[:, z-min_z : z-min_z+1, ...], boxMx, stride_min, out_box, 'lanczos', parallelism=1)[0, 0, ...]
  if z % chunk_size == chunk_size - 1:
      print(datetime.now(), 'writing chunk plane', z//chunk_size)
      data.write_warp_planes(fid, warped, z-chunk_size+1, z+1)
  elif z == max_z:
      print(datetime.now(), 'writing last chunk plane')
      data.write_warp_planes(fid, warped[:z % chunk_size + 1],
                             z//chunk_size * chunk_size, z+1)
