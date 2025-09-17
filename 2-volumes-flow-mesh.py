#!/usr/bin/env python

# takes a pair of overlapping volumes and aligns them

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this first part just does the GPU intensive stuff

# usage: ./2-volumes-flow-mesh.py <patch-size> <stride> <batch-size>

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima.mesh import elastic_mesh_3d

import importlib

data_loader, patch_size, stride, batch_size = sys.argv[1:]
patch_size = int(patch_size)
stride = int(stride)
batch_size = int(batch_size)

print("data_loader =", data_loader)
print("patch_size =", patch_size)
print("stride =", stride)
print("batch_size =", batch_size)

data = importlib.import_module(os.path.basename(data_loader))

ttop, tbot = data.load_data()

#calculate the flow fields

# uses GPU
mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
t0 = time.time()
flow = mfc.flow_field(ttop, tbot,
                      (patch_size, patch_size, patch_size), (stride, stride, stride),
                      batch_size=batch_size)
print("flow_field took", time.time() - t0, "sec")

# the first three channels store the XYZ components of the flow vector, and the
# two remaining channels are measures of estimation quality (see
# sofima.flow_field._batched_peaks for more info)


# Pad to account for the edges of the images where there is insufficient
# context to estimate flow.
pad = patch_size // 2 // stride
flow = np.pad(flow, [[0, 0], [pad, pad], [pad, pad], [pad, pad]], constant_values=np.nan)

# remove uncertain flow estimates by replacing them with NaNs
t0 = time.time()
flow_clean = flow_utils.clean_flow(flow, min_peak_ratio=1.6, min_peak_sharpness=1.6,
                                   max_magnitude=80, max_deviation=20, dim=3)
print("clean_flow took", time.time() - t0, "sec")

### multi-resolution flow fields would be merged here

# find a configuration of the imagery that is compatible with the estimated
# flow field and preserves the original geometry as much as possible.
config = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.01, k=0.1,
                                stride=(stride, stride, stride),
                                num_iters=1000, max_iters=100000,
                                stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                final_cap=10, prefer_orig_order=True)

solved = np.zeros_like(flow_clean)

# also uses GPU
t0 = time.time()
solved, e_kin, num_steps = mesh.relax_mesh(solved, flow_clean, config,
                                           mesh_force=elastic_mesh_3d)
print("relax_mesh took", time.time() - t0, "sec")

params = '.patch'+str(patch_size)+'.stride'+str(stride)

data.save_flow_mesh(flow_clean, solved, params)
