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
    "chunkxy",
    type=int,
    help="of the zarr output",
)
parser.add_argument(
    "chunkz",
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
chunkxy = args.chunkxy
chunkz = args.chunkz

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
print("chunkxy =", chunkxy)
print("chunkz =", chunkz)

data = importlib.import_module(os.path.basename(data_loader))

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+args.scales.replace(",",'')+'.k0'+str(k0)+'.k'+str(k)+'.reps'+str(reps)
invmap = data.load_invmap(basepath, params)

boxMx = bounding_box.BoundingBox(start=(0, 0, 0), size=(invmap.shape[-1], invmap.shape[-2], 1))

print(datetime.now(), 'warping planes')

s_min = min(scales_int)
stride_min = stride * (2**s_min)

warped0 = data.load_data(basepath, min_z,0)
warped = np.zeros((chunkz, *warped0.shape), dtype=warped0.dtype)
curr = np.zeros((1, chunkz, *warped0.shape), dtype=warped0.dtype)
fid = data.open_warp([max_z+1, *warped0.shape], chunkxy, chunkz, basepath, params)

z = min_z
while z <= max_z:
  z0 = z
  print(datetime.now(), 'loading chunk plane', z//chunkz)
  while z <= max_z:
      print(datetime.now(), 'z =', z)
      if z == min_z:
          warped[z % chunkz,:,:] = warped0
      else:
          curr[:,z % chunkz,:,:] = np.transpose(np.expand_dims(np.expand_dims(
                           data.load_data(basepath, z,0), axis=-1), axis=-1), [3, 2, 0, 1])
      z += 1
      if z % chunkz == 0:
          break

  if z > min_z+1:
      sz = [*warped.shape[1:][::-1], z-z0-(z0==min_z)]
      data_box = bounding_box.BoundingBox(start=(0, 0, 0), size=sz)
      out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=sz)

      print(datetime.now(), 'warping chunk plane', (z-1)//chunkz)
      warped[(z0 + (z0==min_z)) % chunkz : (z-1) % chunkz + 1, ...] = warp.warp_subvolume(
          curr[:,(z0 + (z0==min_z)) % chunkz : (z-1) % chunkz + 1, ...], data_box,
          invmap[:, z0-min_z + (z0==min_z) : z-min_z+1, ...], boxMx, stride_min, out_box, 'lanczos',
          parallelism=chunkz)[0, ...]

  print(datetime.now(), 'writing chunk plane', (z-1)//chunkz)
  data.write_warp_planes(fid, warped[z0 % chunkz : (z-1) % chunkz + 1, ...], z0, z)
