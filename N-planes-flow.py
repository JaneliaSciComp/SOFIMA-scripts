#!/usr/bin/env python

# takes a bunch of slices and aligns them

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this first part just does the GPU intensive stuff

# usage: ./N-planes-flow.py <data-loader> <basepath> <min-z> <max-z> <patch-size> <stride> <scales> <batch-size>

import sys
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

from skimage.transform import downscale_local_mean

import importlib

data_loader, basepath, min_z, max_z, patch_size, stride, scales_str, batch_size = sys.argv[1:]
min_z = int(min_z)
max_z = int(max_z)
patch_size = int(patch_size)
stride = int(stride)
scales = [int(x) for x in scales_str.split(',')]
batch_size = int(batch_size)

print("data_loader =", data_loader)
print("basepath =", basepath)
print("min_z =", min_z)
print("max_z =", max_z)
print("patch_size =", patch_size)
print("stride =", stride)
print("scales =", scales_str)
print("batch_size =", batch_size)

data = importlib.import_module(os.path.basename(data_loader))

filenames_noext = data.get_tile_list(min_z, max_z)

def _compute_flow(scales):
  mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
  flows = {s:[] for s in scales}
  _prev = data.load_data(basepath, filenames_noext, 0, 0)
  prev = {s:downscale_local_mean(_prev, (2**s,2**s)) for s in scales}

  fs = []
  with futures.ThreadPoolExecutor() as tpe:
    # Prefetch the next sections to memory so that we don't have to wait for them
    # to load when the GPU becomes available.
    for z in range(1, len(filenames_noext)):
      fs.append(tpe.submit(lambda z=z: data.load_data(basepath, filenames_noext, z,0)))

    fs = fs[::-1]

    for z in range(1,len(filenames_noext)):
      print(datetime.now(), 'z =', z)
      _curr = fs.pop().result()
      curr = {s:downscale_local_mean(_curr, (2**s,2**s)) for s in scales}

      # The batch size is a parameter which impacts the efficiency of the computation (but
      # not its result). It has to be large enough for the computation to fully utilize the
      # available GPU capacity, but small enough so that the batch fits in GPU RAM.
      for s in scales:
          flows[s].append(mfc.flow_field(prev[s], curr[s], (patch_size, patch_size),
                                         (stride, stride), batch_size=batch_size))
      prev = curr

  return flows

print(datetime.now(), 'computing flow')
fNx = _compute_flow(scales)

fN = {}
for s in scales:
    # Convert to [channels, z, y, x].
    flows = np.transpose(np.array(fNx[s]), [1, 0, 2, 3])

    # Pad to account for the edges of the images where there is insufficient context to estimate flow.
    pad = patch_size // 2 // stride
    flows = np.pad(flows, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    fN[s] = flow_utils.clean_flow(flows, min_peak_ratio=1.6, min_peak_sharpness=1.6,
                                  max_magnitude=80, max_deviation=20)

'''
f, ax = plt.subplots(2, 4, figsize=(16, 8))
ax[0,0].hist(flows1x[0, 0, ...][~np.isnan(flows1x[0, 0, ...])])
ax[0,1].hist(f1[0, 0, ...][~np.isnan(f1[0, 0, ...])])
ax[1,0].hist(flows2x[0, 0, ...][~np.isnan(flows2x[0, 0, ...])])
ax[1,1].hist(f2[0, 0, ...][~np.isnan(f2[0, 0, ...])])
ax[0,2].hist(flows1x[1, 0, ...][~np.isnan(flows1x[1, 0, ...])])
ax[0,3].hist(f1[1, 0, ...][~np.isnan(f1[1, 0, ...])])
ax[1,2].hist(flows2x[1, 0, ...][~np.isnan(flows2x[1, 0, ...])])
ax[1,3].hist(f2[1, 0, ...][~np.isnan(f2[1, 0, ...])])
plt.tight_layout()
plt.savefig("flows-f-hist.tif", dpi=100)
'''

'''
# Plot the horizontal component of the flow vector, before (left) and after (right) filtering
f, ax = plt.subplots(2, 4, figsize=(16, 8))
ax[0,0].imshow(flows1x[0, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0,0].title.set_text('H flows1x')
ax[0,1].imshow(f1[0, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0,1].title.set_text('H f1')
ax[1,0].imshow(flows2x[0, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1,0].title.set_text('H flows2x')
ax[1,1].imshow(f2[0, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1,1].title.set_text('H f2')
ax[0,2].imshow(flows1x[1, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0,2].title.set_text('V flows1x')
ax[0,3].imshow(f1[1, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0,3].title.set_text('V f1')
ax[1,2].imshow(flows2x[1, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1,2].title.set_text('V flows2x')
ax[1,3].imshow(f2[1, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1,3].title.set_text('V f2')
plt.tight_layout()
plt.savefig("flows-f.tif", dpi=600)
'''

print(datetime.now(), 'resampling maps')

from scipy import interpolate

fN_hires = {}
s_min = min(scales)
scale_min = 1 / (2**s_min)
boxMx = bounding_box.BoundingBox(start=(0, 0, 0),
                                 size=(fN[s_min].shape[-1], fN[s_min].shape[-2], 1))
for s in scales:
    if s==0:
      fN_hires[0] = fN[0]
      continue

    scale = 1 / (2**s)
    boxNx = bounding_box.BoundingBox(start=(0, 0, 0),
                                     size=(fN[s].shape[-1], fN[s].shape[-2], 1))

    for z in range(fN[s].shape[1]):
      print(datetime.now(), 's =', s, ', z =', z)
      # Upsample and scale spatial components.
      resampled = map_utils.resample_map(
          fN[s][:, z:z + 1, ...],
          boxNx, boxMx, 1 / scale, 1 / scale_min)
      if s not in fN_hires:
          fN_hires[s] = np.zeros((resampled.shape[0], fN[s].shape[1], *resampled.shape[2:]))
      fN_hires[s][:, z:z + 1, ...] = resampled / scale

final_flow = flow_utils.reconcile_flows(tuple(fN_hires[k] for k in scales),
        max_gradient=0, max_deviation=20, min_patch_size=400)

'''
# Plot (left to right): high res. flow, upsampled low res. flow, combined flow to use for alignment.
f, ax = plt.subplots(2, 3, figsize=(7.5, 5))
ax[0,0].imshow(f1[0, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0,0].title.set_text('H f1')
ax[0,1].imshow(f2_hires[0, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0,1].title.set_text('H f2hi')
ax[0,2].imshow(final_flow[0, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0,2].title.set_text('H final')
ax[1,0].imshow(f1[1, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1,0].title.set_text('V f1')
ax[1,1].imshow(f2_hires[1, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1,1].title.set_text('V f2hi')
ax[1,2].imshow(final_flow[1, 0, ...], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1,2].title.set_text('V final')
plt.tight_layout()
plt.savefig("flows-hi-final-b.tif", dpi=300)
'''

'''
f, ax = plt.subplots(1, 2, figsize=(7.5, 5))
ax[0].imshow(f2[0, 0, 300:400, 0:100], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[0].title.set_text('H f2')
ax[1].imshow(f2_hires[0, 0, 600:800, 0:200], cmap=plt.cm.RdBu, vmin=-10, vmax=10)
ax[1].title.set_text('H f2hi')
plt.tight_layout()
plt.savefig("flows-f2-f2hi-d.tif", dpi=300)
'''

params = 'minz'+str(min_z)+'.maxz'+str(max_z)+'.patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+str(scales_str).replace(",",'')

data.save_flow(final_flow, basepath, params)
