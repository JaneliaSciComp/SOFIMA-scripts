#!/usr/bin/env python

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb

# this second part only uses the CPU (and also a lot of RAM).  it depends on the
# output of 2-planes-flow-mesh.py

# ./2-planes-invmap.py data-hess-2-planes /nrs/hess/data/hess_wafers_60_61/export/zarr_datasets/surface-align/run_20251219_110000/pass03-scale2 flat-w61_serial_080_to_089-w61_s080_r00-top-face.zarr 160 8

import os
import argparse
import time
import importlib

from connectomics.common import bounding_box
from sofima import map_utils


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="CPU-intensive inverse mapping - depends on output of 2-planes-flow-mesh.py"
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
parser.add_argument(
    "parallelism",
    type=int,
    help="how many processes/threads to use"
)

args = parser.parse_args()

data_loader = args.data_loader
basepath = args.basepath
top = args.top
patch_size = args.patch_size
stride = args.stride
chunk = args.chunk
parallelism = args.parallelism

print("data_loader =", data_loader)
print("basepath =", basepath)
print("top =", top)
print("patch_size =", patch_size)
print("stride =", stride)
print("chunk =", chunk)
print("parallelism =", parallelism)

data = importlib.import_module(os.path.basename(data_loader))

params = 'patch'+str(patch_size)+'.stride'+str(stride)+'.top'+os.path.splitext(top)[0]

flow = data.load_flow(basepath, params)
mesh = data.load_mesh(basepath, params)

box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

# Image warping requires an inverse coordinate map
# does NOT use GPU, but does use a lot of RAM
t0 = time.time()
invmap = map_utils.invert_map(mesh, box1x, box1x, stride, parallelism=parallelism)
print("invert_map took", time.time() - t0, "sec")

data.save_invmap(chunk, basepath, params, invmap)
