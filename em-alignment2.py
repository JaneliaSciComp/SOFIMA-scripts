#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this second part only uses the CPU (and also a lot of RAM).  it depends on the
# output of em-alignment1.py

# usage : ./em-alignment2.py <patch-size> <stride>

import os
import sys
from concurrent import futures
import time

import jax
import jax.numpy as jnp
import numpy as np
#import matplotlib.pyplot as plt

from connectomics.common import bounding_box
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp

patch_size, stride = sys.argv[1:]
patch_size = int(patch_size)
stride = int(stride)

print("patch_size =", patch_size)
print("stride =", stride)

zbase = "/nrs/hess/data/hess_wafers_60_61/export/hess_wafers_60_61.n5/render/w60_serial_360_to_369/w60_s360_r00_d20_gc_align_b_ic/s0"

import tensorstore as ts

params = '.patch'+str(patch_size)+'.stride'+str(stride)

flow = ts.open({
    'driver': 'zarr',
    'kvstore': {"driver":"file", "path":os.path.join(zbase, '1.flow'+params+'.zarr')},
    }).result()
solved = ts.open({
    'driver': 'zarr',
    'kvstore': {"driver":"file", "path":os.path.join(zbase, '2.solved-mesh'+params+'.zarr')},
    }).result()

solved = np.array(solved.read().result())

box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

#fig, axs = plt.subplots(1, 3, figsize=(10, 4))
#axs[0].imshow(solved[1][1,:,:])


# Image warping requires an inverse coordinate map
# does NOT use GPU, but does use a lot of RAM
t0 = time.time()
inv_map = map_utils.invert_map(solved, box1x, box1x, stride)
print("invert_map took", time.time() - t0, "sec")


#fig, axs = plt.subplots(1, 3, figsize=(10, 4))
#axs[0].imshow(inv_map[0,1,:,:])

#warped = [ttop[0:30000,0:30000].read().result()[np.newaxis, :]]
#
#
#data_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(30000, 30000, 1))
#out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(30000, 30000, 1))
#data = tbot[data_box.start[0]:data_box.end[0],
#            data_box.start[1]:data_box.end[1]].read().result()
#data = data[np.newaxis, np.newaxis, :]
#
#warped.append(warp.warp_subvolume(data, data_box, inv_map[:, 1:2, ...], box1x, stride, out_box, 'lanczos', parallelism=1)[0, ...])
#
#warped_xyz = np.transpose(np.concatenate(warped, axis=0), [2, 1, 0])

#orig_top = np.zeros((warped_xyz.shape[0], warped_xyz.shape[1], 3), dtype=np.float32)
#orig_top[:, :, 0] = warped_xyz[:,:,0]
#orig_top[:, :, 1] = warped_xyz[:,:,0]
#orig_top /= 255
#
#orig_bot = np.zeros((warped_xyz.shape[0], warped_xyz.shape[1], 3), dtype=np.float32)
#orig_bot[:, :, 2] = np.transpose(ttop[19900:20000, 19900:20000].read().result(), [1,0])
#orig_bot /= 255
#
#warp_bot = np.zeros((warped_xyz.shape[0], warped_xyz.shape[1], 3), dtype=np.float32)
#warp_bot[:, :, 2] = warped_xyz[:,:,1]
#warp_bot /= 255
#
#orig_ovr = np.zeros((warped_xyz.shape[0], warped_xyz.shape[1], 3), dtype=np.float32)
#orig_ovr[:, :, 0] = warped_xyz[:,:,0]
#orig_ovr[:, :, 1] = warped_xyz[:,:,0]
#orig_ovr[:, :, 2] = np.transpose(ttop[19900:20000, 19900:20000].read().result(), [1,0])
#orig_ovr /= 255
#
#warp_ovr = np.zeros((warped_xyz.shape[0], warped_xyz.shape[1], 3), dtype=np.float32)
#warp_ovr[:, :, 0] = warped_xyz[:,:,0]
#warp_ovr[:, :, 1] = warped_xyz[:,:,0]
#warp_ovr[:, :, 2] = warped_xyz[:,:,1]
#warp_ovr /= 255
#
#fig, axs = plt.subplots(2, 3, figsize=(10, 4))
#axs[0,0].imshow(orig_bot, vmin=0, vmax=1)
#axs[0,1].imshow(orig_top, vmin=0, vmax=1)
#axs[0,2].imshow(orig_ovr, vmin=0, vmax=1)
#axs[1,1].imshow(warp_top, vmin=0, vmax=1)
#axs[1,2].imshow(warp_ovr, vmin=0, vmax=1)


#tbot_warped = ts.open({
#    'driver': 'zarr',
#    'kvstore': {"driver":"file", "path":os.path.join(zbase, 'warped-'+zbot)},
#    'metadata': {
#        "compressor":{"id":"zstd","level":3},
#        "shape":[30000,30000],
#        "fill_value":0,
#        'dtype': '|u1',
#    },
#    'create': True,
#    'delete_existing': True,
#    }).result()
#
#tbot_warped[0:30000,0:30000].write(warped_xyz[:,:,1]).result()
#
#tbot_notwarped = ts.open({
#    'driver': 'zarr',
#    'kvstore': {"driver":"file", "path":os.path.join(zbase, 'notwarped-'+zbot)},
#    'metadata': {
#        "compressor":{"id":"zstd","level":3},
#        "shape":[30000,30000],
#        "fill_value":0,
#        'dtype': '|u1',
#    },
#    'create': True,
#    'delete_existing': True,
#    }).result()
#
#tbot_notwarped[0:30000,0:30000].write(np.transpose(tbot[0:30000, 0:30000].read().result(), [1,0])).result()
#
#ttop_notwarped = ts.open({
#    'driver': 'zarr',
#    'kvstore': {"driver":"file", "path":os.path.join(zbase, 'notwarped-'+ztop)},
#    'metadata': {
#        "compressor":{"id":"zstd","level":3},
#        "shape":[30000,30000],
#        "fill_value":0,
#        'dtype': '|u1',
#    },
#    'create': True,
#    'delete_existing': True,
#    }).result()
#
#ttop_notwarped[0:30000,0:30000].write(np.transpose(ttop[0:30000, 0:30000].read().result(), [1,0])).result()

#fig, axs = plt.subplots(1, 3, figsize=(10, 4))
#axs[0].imshow(warped_xyz[0:-1:100,0:-1:100,0])
#axs[1].imshow(warped_xyz[0:-1:100,0:-1:100,1])

invmap_zarr = ts.open({
    'driver': 'zarr',
    'kvstore': {"driver":"file", "path":os.path.join(zbase, '3.invmap'+params+'.zarr')},
    'metadata': {
        "compressor":{"id":"zstd","level":3},
        "shape":inv_map.shape,
        "fill_value":0,
        'dtype': '<f8',
    },
    'create': True,
    'delete_existing': True,
    }).result()

invmap_zarr.write(inv_map).result()
