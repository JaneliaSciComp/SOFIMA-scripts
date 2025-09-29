#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this third part only uses the CPU (and also a lot of RAM).  it depends on the
# output of N-planes-invmap.py

# usage : ./N-planes-warp.py <data-loader> <basepath> <min-z> <max-z> <patch-size> <stride> <scales>

import sys
import os
import numpy as np
from connectomics.common import bounding_box
from sofima import warp
from datetime import datetime

import importlib

data_loader, basepath, min_z, max_z, patch_size, stride, scales_str = sys.argv[1:]
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

data = importlib.import_module(os.path.basename(data_loader))

filenames_noext = data.get_tile_list(min_z, max_z)

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+str(scales_str).replace(",",'')
flow = data.load_mesh(basepath, params)
invmap = data.load_invmap(basepath, params)

boxMx = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

print(datetime.now(), 'warping planes')

chunk_size = 128

s_min = min(scales)
stride_min = stride * (2**s_min)

warped0 = data.load_data(basepath, filenames_noext, 0,0)
warped = np.empty((chunk_size, *warped0.shape), dtype=warped0.dtype)
warped[0,...] = warped0
fid = data.open_warp([len(filenames_noext), *warped0.shape], chunk_size, basepath, params)

futures = []

for z in range(1, len(filenames_noext)):
  print(datetime.now(), 'z =', z)
  data_box = bounding_box.BoundingBox(start=(0, 0, 0), size=[*np.flip(np.array(warped0.shape)),1])
  out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=[*np.flip(np.array(warped0.shape)),1])

  curr = np.transpose(np.expand_dims(np.expand_dims(data.load_data(basepath, filenames_noext, z,0),
                                                    axis=-1),
                                     axis=-1),
                      [3, 2, 0, 1])
  warped[z % chunk_size, ...] = warp.warp_subvolume(curr, data_box,
      invmap[:, z:z+1, ...], boxMx, stride_min, out_box, 'lanczos', parallelism=1)[0, 0, ...]
  if z % chunk_size == chunk_size - 1:
      futures.append(data.write_warp_planes(fid, warped, z-chunk_size+1, z+1))
  elif z == len(filenames_noext) - 1:
      futures.append(data.write_warp_planes(fid, warped[:z % chunk_size + 1],
                                            z//chunk_size * chunk_size, z+1))

print(datetime.now(), 'waiting for writing to finish')
while len(futures)>0:
  print(datetime.now(), len(futures), 'chunk planes remaining')
  futures.pop(0).result()
