#!/usr/bin/env python

# takes a bunch of slices and aligns them

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this second part just does the GPU intensive stuff.  it depends on the
# output of N-planes-flow.py

# usage: ./N-planes-mesh.py <data-loader> <min-z> <max-z> <patch-size> <stride> <batch-size>
# bsub -Pcellmap -n4 -gpu "num=1" -q gpu_l4 -Is /bin/zsh

import os
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

data_loader, min_z, max_z, patch_size, stride, batch_size = sys.argv[1:]
min_z = int(min_z)
max_z = int(max_z)
patch_size = int(patch_size)
stride = int(stride)
batch_size = int(batch_size)

print("data_loader =", data_loader)
print("min_z =", min_z)
print("max_z =", max_z)
print("patch_size =", patch_size)
print("stride =", stride)
print("batch_size =", batch_size)

data = importlib.import_module(os.path.basename(data_loader))

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)

flow = data.load_flow(params)

config = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.01, k=0.1,
                                stride=(stride, stride),
                                num_iters=1000, max_iters=100000,
                                stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                final_cap=10, prefer_orig_order=True)

solved = [np.zeros_like(flow[:, 0:1, ...])]
origin = jnp.array([0., 0.])

print(datetime.now(), 'composing maps')
for z in range(0, flow.shape[1]):
  print(datetime.now(), 'z =', z)
  prev = map_utils.compose_maps_fast(flow[:, z:z+1, ...], origin, stride,
                                     solved[-1], origin, stride)
  x = np.zeros_like(solved[0])
  x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
  x = np.array(x)
  solved.append(x)

solved = np.concatenate(solved, axis=1)

data.save_mesh(solved, params)
