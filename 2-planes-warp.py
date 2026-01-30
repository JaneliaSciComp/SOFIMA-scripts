#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this third part only uses the CPU (and also a lot of RAM).  it depends on the
# output of 2-planes-invmap.py

# bsub -Phess -n8 -Is /bin/zsh
# conda run -n multi-sem --no-capture-output python ./2-planes-warp.py data-hess-2-planes /nrs/hess/data/hess_wafers_60_61/export/zarr_datasets/surface-align/run_20251219_110000/pass03-scale2 flat-w61_serial_080_to_089-w61_s080_r00-top-face.zarr flat-w61_serial_070_to_079-w61_s079_r00-bot-face.zarr 160 8 1024

import sys
import os
import argparse
import numpy as np
from connectomics.common import bounding_box
from sofima import warp
from datetime import datetime

import importlib

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="warps a slice according to a coordinate map"
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
    "top",
    help="filename of top of one slab"
)
parser.add_argument(
    "bot",
    help="filename of bottom of an adjacent slab"
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
    "chunk",
    type=int,
    help="of the zarr output",
)

args = parser.parse_args()

data_loader = args.data_loader
basepath = args.basepath
top = args.top
bot = args.bot
patch_size = args.patch_size
stride = args.stride
chunk = args.chunk

print("data_loader =", data_loader)
print("basepath =", basepath)
print("top =", top)
print("bot =", bot)
print("patch_size =", patch_size)
print("stride =", stride)
print("chunk =", chunk)

data = importlib.import_module(os.path.basename(data_loader))

params = 'patch'+str(patch_size)+'.stride'+str(stride)+'.top'+top

invmap = data.load_invmap(basepath, params)

print(datetime.now(), 'loading data')
ttop, tbot = data.load_data(basepath, top, bot)
warped = np.zeros((2, *tbot.shape), dtype=tbot.dtype)
curr = np.zeros((1, 1, *tbot.shape), dtype=tbot.dtype)

warped[0,...] = tbot
curr[0,0,...] = ttop

sz = [*warped.shape[1:][::-1], 1]
data_box = bounding_box.BoundingBox(start=(0, 0, 0), size=sz)
out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=sz)
boxMx = bounding_box.BoundingBox(start=(0, 0, 0),
                               size=(invmap.shape[-1], invmap.shape[-2], 1))

print(datetime.now(), 'warping data')
warped[1, ...] = warp.warp_subvolume(
        curr, data_box, invmap, boxMx, stride, out_box, 'lanczos')[0, ...]

print(datetime.now(), 'writing data')
data.write_warp(chunk, basepath, params, warped)
