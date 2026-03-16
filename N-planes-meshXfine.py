#!/usr/bin/env python

# takes a bunch of slices and aligns them

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this second part just does the GPU intensive stuff.  it depends on the
# output of N-planes-flow.py

import sys
import os
import argparse
from concurrent import futures
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

from connectomics.common import bounding_box
from connectomics.volume import subvolume
from sofima.processor import maps
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from datetime import datetime 


import importlib

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="relaxes the springed mesh using the flow field - GPU intensive"
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
    "minz",
    type=int,
    help="lower bound on the planes to align"
)
parser.add_argument(
    "maxz",
    type=int,
    help="upper bound on the planes to align"
)
parser.add_argument(
    "patch_size",
    type=str,
    help="Side length of (square) patch for processing (in pixels, e.g., 32)",
)
parser.add_argument(
    "stride",
    type=str,
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
    "nslices",
    type=int,
    help="the size the each block"
)

args = parser.parse_args()

data_loader = args.data_loader
basepath = args.basepath
minz = args.minz
maxz = args.maxz
patch_size = args.patch_size
stride = args.stride
scales = args.scales
k0 = args.k0
k = args.k
nslices = args.nslices

print("data_loader =", data_loader)
print("basepath =", basepath)
print("minz =", minz)
print("maxz =", maxz)
print("patch_size =", patch_size)
print("stride =", stride)
print("scales =", scales)
print("k0 =", k0)
print("k =", k)
print("nslices =", nslices)

stride_int_min = [int(x) for x in args.stride.split(',')][-1]

data = importlib.import_module(os.path.basename(data_loader))

params = 'patch'+patch_size+'.stride'+stride+'.scales'+args.scales.replace(",",'')+'.k0'+str(k0)+'.k'+str(k)

print(datetime.now(), 'loading meshXcoarse')
xblk_upsampled = data.load_mesh_coarse(basepath, params, "Xcoarse")

print(datetime.now(), 'loading meshXcoarse_inv')
xblk_inv = data.load_mesh_coarse(basepath, params, "Xcoarse_inv")

print(datetime.now(), 'loading main and inverted main')
main_shape = (xblk_upsampled.shape[0], maxz - minz + 1, *xblk_upsampled.shape[2:])
main = np.zeros(main_shape, dtype=xblk_upsampled.dtype)
main_inv = np.zeros(main_shape, dtype=xblk_upsampled.dtype)

minz0 = (minz // nslices) * nslices
for z in range(minz0, maxz + 1, nslices):
    minz2 = max(z, minz)
    maxz2 = min(z + nslices - 1, maxz)
    istart = minz2 + 1 - minz
    iend = maxz2 + 1 - minz
    main[:, istart:iend, ...] = data.load_mesh(basepath, params, minz2+1, maxz2, "")
    main_inv[:, istart:iend, ...] = data.load_invmap(basepath, params, minz2+1, maxz2, False)

print(datetime.now(), 'loading inverted last')
last_inv = np.zeros((main.shape[0], maxz - minz + 1, *main.shape[2:]), dtype=main.dtype)

z_map = {}
minz0 = (minz // nslices) * nslices
for iz, z in enumerate(range(minz0, maxz + 1, nslices)):
    minz2 = max(z, minz)
    maxz2 = min(z + nslices - 1, maxz - 1)
    itarget = maxz2 - minz + 1
    z_map[str(itarget)] = iz+1
    last_inv[:, itarget:itarget+1, ...] = data.load_invmap(basepath, params, maxz2+1, maxz2+1, False)

class ReconcileCrossBlockMaps(maps.ReconcileCrossBlockMaps):
  def _open_volume(self, path: str):
    if path == 'main_inv':
      return main_inv
    elif path == 'last_inv':
      return last_inv
    elif path == 'xblk':
      return xblk_upsampled
    elif path == 'xblk_inv':
      return xblk_inv
    else:
      raise ValueError(f'Unknown volume {path}')

config = maps.ReconcileCrossBlockMaps.Config(
    cross_block='xblk',
    cross_block_inv='xblk_inv',
    last_inv='last_inv',
    main_inv='main_inv',
    z_map=z_map,
    stride=stride_int_min,
    xy_overlap=0)

print(datetime.now(), 'reconciling')
reconcile = ReconcileCrossBlockMaps(config)
reconcile.set_effective_subvol_and_overlap(xblk_upsampled.shape[1:][::-1], (0, 0, 0))
main_box = bounding_box.BoundingBox(start=(0, 0, 0),
                                    size=(*main.shape[2:][::-1], maxz-minz+1))

global_map = reconcile.process(subvolume.Subvolume(main, main_box), verbose=True)

print(datetime.now(), 'writing meshX')
fid = data.create_mesh(global_map.shape, basepath, params, 1, "Xfine")
for z in range(minz, maxz+1):
    data.write_mesh_plane(fid, global_map.data[:,z-minz:z-minz+1,:,:], z)

print(datetime.now(), 'finished!')
