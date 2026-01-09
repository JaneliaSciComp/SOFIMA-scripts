#!/usr/bin/env python

# takes a bunch of slices and aligns them

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this first part just does the GPU intensive stuff

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

from skimage.transform import downscale_local_mean

import importlib

debug = False  # save intermediate steps

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="computes the flow fields between a pair of slices - GPU intensive"
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
    "batch_size",
    type=int,
    help="how many patches to process simultaneously",
)
parser.add_argument(
    "write_metadata",
    type=int,
    help="whether to write the zarr metadata for not"
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
batch_size = args.batch_size
write_metadata = args.write_metadata

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
print("batch_size =", batch_size)
print("write_metadata =", write_metadata)

data = importlib.import_module(os.path.basename(data_loader))

def _compute_flow(scales, prev_flows=None):
  mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
  flows = {s:[] for s in scales}
  _prev = data.load_data(basepath, min_z, 0)
  prev = {s:_prev if s==0 else downscale_local_mean(_prev, (2**s,2**s)) for s in scales}
  if 0 not in scales:  del _prev

  fs = []
  with futures.ThreadPoolExecutor() as tpe:
    # Prefetch the next sections to memory so that we don't have to wait for them
    # to load when the GPU becomes available.
    for z in range(min_z+1, max_z+1):
      fs.append(tpe.submit(lambda z=z: data.load_data(basepath, z,0)))

    fs = fs[::-1]

    for z in range(min_z+1, max_z+1):
      print(datetime.now(), 'z =', z)
      _curr = fs.pop().result()
      curr = {s:_curr if s==0 else downscale_local_mean(_curr, (2**s,2**s)) for s in scales}
      if 0 not in scales:  del _curr

      # The batch size is a parameter which impacts the efficiency of the computation (but
      # not its result). It has to be large enough for the computation to fully utilize the
      # available GPU capacity, but small enough so that the batch fits in GPU RAM.
      for s in scales:
          flows[s].append(mfc.flow_field(prev[s], curr[s],
                                         (patch_size, patch_size),
                                         (stride, stride),
                                         batch_size = batch_size,
                                         pre_targeting_field = prev_flows[s][z-(min_z+1)][:2, ::] if prev_flows else None,
                                         pre_targeting_step = (stride, stride)))
      prev = curr

  return flows

print(datetime.now(), 'computing flow')
fNx = _compute_flow(scales_int)
for _ in range(reps-1):
    fNx = _compute_flow(scales_int, fNx)

fN = {}
for s in scales_int:
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
s_min = min(scales_int)
scale_min = 1 / (2**s_min)
boxMx = bounding_box.BoundingBox(start=(0, 0, 0),
                                 size=(fN[s_min].shape[-1], fN[s_min].shape[-2], 1))
for s in scales_int:
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

final_flow = flow_utils.reconcile_flows(tuple(fN_hires[k] for k in scales_int),
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

params = 'patch'+str(patch_size)+'.stride'+str(stride)+'.scales'+args.scales.replace(",",'')+'.k0'+str(k0)+'.k'+str(k)+'.reps'+str(reps)

data.save_flow(final_flow, min_z, max_z, basepath, params, write_metadata)

if debug:
    for s in scales_int:
        flows = np.transpose(np.array(fNx[s]), [1, 0, 2, 3])
        flows = np.pad(flows, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
        np.save(os.path.join(basepath, 'fNx.s'+str(s)+'.'+params+'.npy'), flows)
        np.save(os.path.join(basepath, 'fN.s'+str(s)+'.'+params+'.npy'), fN[s])
        np.save(os.path.join(basepath, 'fN_hires.s'+str(s)+'.'+params+'.npy'), fN_hires[s])
