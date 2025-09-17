#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# generate overlapped images to test that everything works

# usage: ./2-planes-test.py <data-loader> <patch-size> <stride>

import os
import numpy as np
from connectomics.common import bounding_box
from sofima import warp
import matplotlib.pyplot as plt

import importlib

data_loader, patch_size, stride = sys.argv[1:]
patch_size = int(patch_size)
stride = int(stride)

print("data_loader =", data_loader)
print("patch_size =", patch_size)
print("stride =", stride)

data = importlib.import_module(os.path.basename(data_loader))

params = '.patch'+str(patch_size)+'.stride'+str(stride)

flow, mesh = data.load_flow_mesh(params)
invmap = data.load_invmap(params)

ttop, tbot = data.load_data()

ttop_crop, tbot_crop, icrop_start, icrop_size = data.crop_data(ttop, tbot)
image_box = bounding_box.BoundingBox(start=(*icrop_start, 1), size=(*icrop_size,1))

ttop_tight_crop, tbot_tight_crop, itight_crop_start, itight_crop_size = data.tight_crop_data(ttop, tbot)
out_box = bounding_box.BoundingBox(start=(*itight_crop_start, 1), size=(*itight_crop_size,1))

warped = [ttop_tight_crop[np.newaxis, :]]

image = np.transpose(tbot_crop[:,:,np.newaxis,np.newaxis], [3,2,0,1])

box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))
warped.append(warp.warp_subvolume(image, image_box,
                                  invmap, box1x, stride,
                                  out_box, 'lanczos', parallelism=1)[0, ...])

img_shape = (warped[0].shape[1], warped[0].shape[2], 3)

def qscale(img, p=(0.1,0.99)):
    q = np.quantile(img, p)
    out = (img-q[0]) / (q[1]-q[0])
    out[out<0] = 0
    out[out>1] = 1
    return out

orig_top = np.zeros(img_shape, dtype=np.float32)
orig_top[:,:,0] = qscale(warped[0])
orig_top[:,:,1] = qscale(warped[0])

orig_bot = np.zeros(img_shape, dtype=np.float32)
orig_bot[:,:,2] = qscale(tbot_tight_crop)

orig_ovr = np.zeros(img_shape, dtype=np.float32)
orig_ovr[:,:,0] = qscale(warped[0])
orig_ovr[:,:,1] = qscale(warped[0])
orig_ovr[:,:,2] = qscale(tbot_tight_crop)

warp_bot = np.zeros(img_shape, dtype=np.float32)
warp_bot[:, :, 2] = qscale(warped[1])

warp_ovr = np.zeros(img_shape, dtype=np.float32)
warp_ovr[:,:,0] = qscale(warped[0])
warp_ovr[:,:,1] = qscale(warped[0])
warp_ovr[:,:,2] = qscale(warped[1])

def set_axis(ax, t):
    ax.set_title(t)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xticklabels([]);  ax.set_yticklabels([]);
    ax.set_xticks([]);  ax.set_yticks([]);

fig, axs = plt.subplots(2, 3, figsize=(6, 8))
axs[0,0].imshow(orig_top, vmin=0, vmax=1)
set_axis(axs[0,0], 'orig top')
axs[0,1].imshow(orig_bot, vmin=0, vmax=1)
set_axis(axs[0,1], 'orig bot')
axs[0,2].imshow(orig_ovr, vmin=0, vmax=1)
set_axis(axs[0,2], 'orig ovr')
ix0 = itight_crop_start[0] // stride
ix1 = (itight_crop_start[0] + itight_crop_size[0]) // stride
iy0 = itight_crop_start[1] // stride
iy1 = (itight_crop_start[1] + itight_crop_size[1]) // stride
axs[1,0].quiver(-np.ma.array(flow[0,0, iy0:iy1, ix0:ix1], mask=np.isnan(flow[0,0, iy0:iy1, ix0:ix1])),
                 np.ma.array(flow[1,0, iy0:iy1, ix0:ix1], mask=np.isnan(flow[1,0, iy0:iy1, ix0:ix1])))
axs[1,0].set_aspect('equal', 'box')
set_axis(axs[1,0], 'flow')
axs[1,1].imshow(warp_bot, vmin=0, vmax=1)
set_axis(axs[1,1], 'warp bot')
axs[1,2].imshow(warp_ovr, vmin=0, vmax=1)
set_axis(axs[1,2], 'warp ovr')
fig.tight_layout()
plt.savefig("overlay.png")
