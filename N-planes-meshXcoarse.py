#!/usr/bin/env python

# takes a bunch of slices and aligns them

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

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

print(datetime.now(), 'loading xblocks')
xblock_flow = []
minz0 = (minz // nslices) * nslices
for z in range(minz0, maxz + 1, nslices):
    minz2 = max(z, minz)
    maxz2 = min(z + nslices - 1, maxz)
    xblock_flow.append(data.load_mesh(basepath, params, maxz2+1, maxz2+1, ""))

xblock_flow = np.concatenate(xblock_flow, axis=1)

map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=xblock_flow.shape[1:][::-1])
map2x_box = map_box.scale(0.5)
xblk_stride = stride_int_min * 2

print(datetime.now(), 'downsampling xblocks')
xblock_flow2 = map_utils.resample_map(xblock_flow, map_box, map2x_box, stride_int_min, xblk_stride,
                                      parallelism=None, verbose=True)

xblk_config = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.001, k=0.1,
                                     stride=(xblk_stride, xblk_stride), num_iters=1000,
                                     max_iters=100000, stop_v_max=0.005, dt_max=1000,
                                     start_cap=0.01, final_cap=10, prefer_orig_order=True)

origin = jnp.array([0., 0.])

print(datetime.now(), 'relaxing xblocks')
xblk = []
for z in range(xblock_flow2.shape[1]):
    print(datetime.now(), 'z =', z)
    if z == 0:
        prev = xblock_flow2[:, z:z+1, ...]
    else:
        prev = map_utils.compose_maps_fast(xblock_flow2[:, z:z+1, ...], origin, xblk_stride, xblk[-1], origin, xblk_stride)
    x = np.zeros_like(xblock_flow2[:, 0:1, ...])
    x, e_kin, num_steps = mesh.relax_mesh(x, prev, xblk_config)
    xblk.append(np.array(x))

xblk = np.concatenate(xblk, axis=1)

print(datetime.now(), 'upsampling xblocks')
xblk_upsampled = map_utils.resample_map(xblk, map2x_box, map_box, stride_int_min * 2, stride_int_min,
                                        parallelism=None, verbose=True)
print(datetime.now(), 'inverting xblocks')
xblk_inv = map_utils.invert_map(xblk_upsampled, map_box, map_box, stride_int_min,
                                parallelism=None, verbose=True)

print(datetime.now(), 'writing meshXcoarse')
data.save_mesh_coarse(xblk_upsampled, basepath, params, "Xcoarse")

print(datetime.now(), 'writing meshXcoarse_inv')
data.save_mesh_coarse(xblk_inv, basepath, params, "Xcoarse_inv")

print(datetime.now(), 'finished!')
