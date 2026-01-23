#!/usr/bin/env python

# takes a bunch of slices and aligns them

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this second part just does the GPU intensive stuff.  it depends on the
# output of N-planes-flow.py

import sys
import os
import argparse
from concurrent import futures
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from connectomics.common import bounding_box
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

params = 'patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+args.scales.replace(",",'')+'.k0'+str(k0)+'.k'+str(k)+'.reps'+str(reps)

flow = data.load_flow(basepath, params, min_z, max_z)

config = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=k0, k=k,
                                stride=(stride, stride),
                                num_iters=1000, max_iters=100000,
                                stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                final_cap=10, prefer_orig_order=True)

solved = np.zeros_like(flow[:, 0:1, ...])
origin = jnp.array([0., 0.])

s_min = min(scales_int)
stride_min = stride * (2**s_min)

write_metadata=1
fid = data.create_mesh(solved.shape, basepath, params, write_metadata)
if write_metadata:
  data.write_mesh_plane(fid, solved, z0)

print(datetime.now(), 'composing maps')
for z in range(min_z+1, max_z+1):
  print(datetime.now(), 'z =', z)
  zf = z - min_z - 1
  prev = map_utils.compose_maps_fast(flow[:, zf:zf+1, ...], origin, stride_min,
                                     solved, origin, stride_min)
  x = np.zeros_like(solved)
  x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
  solved = np.array(x)
  data.write_mesh_plane(fid, solved, z)
