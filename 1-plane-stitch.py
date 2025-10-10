#!/usr/bin/env python

# takes tiles in a plane and stitches them together

# a derivative of https://github.com/google-research/sofima/blob/main/notebooks/em_stitching.ipynb

# a GPU and 1 core needed

import sys
import os
import argparse
import functools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import importlib

debug = False  # save unstitched but warped tiles

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Takes a pair of slices and aligns them - GPU intensive processing"
)
parser.add_argument(
    "data_loader",
    help="Data loader module name, e.g., data-test-2-planes"
)
parser.add_argument(
    "z",
    type=int,
    help="slice of interest"
)
parser.add_argument(
    "level",
    type=int,
    help="the spatial scale / resolution to stitch together"
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
    "outpath",
    help="path to save the results"
)
parser.add_argument(
    "write_metadata",
    type=int,
    help="whether to write the zarr metadata for not"
)
parser.add_argument(
    "chunk_size",
    type=int,
    help="of the zarr output",
)

args = parser.parse_args()

data_loader = args.data_loader
z = args.z
level = args.level
patch_size = args.patch_size
stride = args.stride
k0 = args.k0
k = args.k
outpath = args.outpath
write_metadata = args.write_metadata
chunk_size = args.chunk_size

print("data_loader =", data_loader)
print("z =", z)
print("level =", level)
print("patch_size =", patch_size)
print("stride =", stride)
print("k0 =", k0)
print("k =", k)
print("outpath =", outpath)
print("write_metadata =", write_metadata)
print("chunk_size =", chunk_size)

data = importlib.import_module(os.path.basename(data_loader))

planepath = data.get_tilepath(z)
tile_map = data.load_data(planepath, level)

from sofima import stitch_rigid
cx, cy = stitch_rigid.compute_coarse_offsets(data.tile_space, tile_map,
            overlaps_xy=((200, 300), (200, 300)), min_overlap=0)

coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy)

from sofima import stitch_elastic

# The stride (in pixels) specifies the resolution at which to compute the flow
# fields between tile pairs. This is the same as the resolution at which the
# mesh is later optimized. The more deformed the tiles initially are, the lower
# the stride needs to be to get good stitching results.
cx = np.squeeze(cx)
cy = np.squeeze(cy)
fine_x, offsets_x = stitch_elastic.compute_flow_map(tile_map, cx, 0,
        stride=(stride, stride), patch_size=(patch_size, patch_size), batch_size=4)  # (x,y) -> (x+1,y)
fine_y, offsets_y = stitch_elastic.compute_flow_map(tile_map, cy, 1,
        stride=(stride, stride), patch_size=(patch_size, patch_size), batch_size=4)  # (x,y) -> (x,y+1)

from sofima import flow_utils

kwargs = {"min_peak_ratio": 1.4, "min_peak_sharpness": 1.4,
          "max_deviation": 5, "max_magnitude": 0}
fine_x = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :]
          for k, v in fine_x.items()}
fine_y = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :]
          for k, v in fine_y.items()}

kwargs = {"min_patch_size": 10, "max_gradient": -1, "max_deviation": -1}
fine_x = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :]
          for k, v in fine_x.items()}
fine_y = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :]
          for k, v in fine_y.items()}

from sofima import mesh

data_x = (cx, fine_x, offsets_x)
data_y = (cy, fine_y, offsets_y)

fx, fy, x, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
    data_x, data_y, list(tile_map.keys()),
    coarse_mesh[:, 0, ...], stride=(stride, stride),
    tile_shape=next(iter(tile_map.values())).shape)

@jax.jit
def prev_fn(x):
  target_fn = ft.partial(stitch_elastic.compute_target_mesh, x=x, fx=fx,
                         fy=fy, stride=(stride, stride))
  x = jax.vmap(target_fn)(nbors)
  return jnp.transpose(x, [1, 0, 2, 3])

# These detault settings are expect to work well in most configurations. Perhaps
# the most salient parameter is the elasticity ratio k0 / k. The larger it gets,
# the more the tiles will be allowed to deform to match their neighbors (in which
# case you might want use aggressive flow filtering to ensure that there are no
# inaccurate flow vectors). Lower ratios will reduce deformation, which, depending
# on the initial state of the tiles, might result in visible seams.
config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=k0, k=k, stride=(stride, stride),
                                num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                dt_max=100, prefer_orig_order=True,
                                start_cap=0.1, final_cap=10., remove_drift=True)

x, ekin, t = mesh.relax_mesh(x, None, config, prev_fn=prev_fn)

from sofima import warp

# Unpack meshes into a dictionary.
idx_to_key = {v: k for k, v in key_to_idx.items()}
meshes = {idx_to_key[i]: np.array(x[:, i:i+1 :, :]) for i in range(x.shape[1])}

# Warp the tiles into a single image.
stitched, _ = warp.render_tiles(tile_map, meshes, stride=(stride, stride))

# Warp the tiles into a single image.
stitched, _ = warp.render_tiles(tile_map, meshes, stride=(stride, stride))

data.save_plane(outpath, z, stitched, level, write_metadata, chunk_size)


if debug:
    stitched = {}
    maxdim = [0,0]
    for k in tile_map.keys():
        tm = {k: tile_map[k]}
        msh = {k: meshes[k]}
        stitched[k], _ = warp.render_tiles(tm, msh, stride=(stride, stride))
        maxdim[0] = max(maxdim[0], stitched[k].shape[0])
        maxdim[1] = max(maxdim[1], stitched[k].shape[1])

    data.save_tiles(outpath, planepath, stitched, maxdim)
