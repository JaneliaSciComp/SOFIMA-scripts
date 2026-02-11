#!/usr/bin/env python

#bsub -Pcellmap -n8 -Is /bin/zsh
#./multiscale.py /nrs/cellmap/arthurb/aphid/final warped.patch128.stride8.scales12.k00.01.k0.1.reps2.zarr pyramid_anisotropic_v2.zarr 4

import argparse
import tensorstore as ts
import numpy as np
import json
import os
import shutil
from datetime import datetime 

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="warps a slice according to a coordinate map"
)
parser.add_argument(
    "basepath",
    help="filepath to stitched planes"
)
parser.add_argument(
    "inpath",
    help="filepath to s0"
)
parser.add_argument(
    "outpath",
    help="filepath to pyramid"
)
parser.add_argument(
    "nlevels",
    type=int,
    help="numbers of scales (beyond s0)"
)
parser.add_argument(
    "chunkxy",
    type=int,
    help="of the zarr output",
)
parser.add_argument(
    "chunkz",
    type=int,
    help="of the zarr output",
)

args = parser.parse_args()

basepath = args.basepath
inpath = args.inpath
outpath = args.outpath
nlevels = args.nlevels
chunkxy = args.chunkxy
chunkz = args.chunkz

print("basepath =", basepath)
print("inpath =", inpath)
print("outpath =", outpath)
print("nlevels =", nlevels)
print("chunkxy =", chunkxy)
print("chunkz =", chunkz)

# --- CONFIGURATION FOR MEMORY SAFETY ---
# Adjust these based on your machine's available RAM
MEMORY_LIMIT_BYTES = 32_000_000_000  # bytes
CONCURRENCY_LIMIT = 16               # processes

def write_ome_zarr_v2_metadata(output_root, num_levels, scale_factors, axes=('z', 'y', 'x')):
    datasets = []
    for level in range(num_levels + 1):
        current_level_scales = [base ** level for base in scale_factors]
        current_level_translations = [x / 2 - 0.5 for x in current_level_scales] 
#1: 1-0, 1-0,    1-0,    1-0
#2: 1-0, 2-0.5,  4-1.5,  8-3.5, 16-7.5
#4: 1-0, 4-1.5, 16-7.5, 32-
#
#123456789abcdefg
#159d
#1
        datasets.append({
            "path": f"s{level}",
            "coordinateTransformations": [
                {"type": "scale", "scale": current_level_scales},
                {"type": "translation", "translation": current_level_translations}]
        })

    metadata = {
        "multiscales": [{
            "version": "0.4",
            "name": "pyramid",
            "axes": [{"name": ax, "type": "space"} for ax in axes],
            "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}],
            "datasets": datasets,
        }]
    }

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, '.zattrs'), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(output_root, '.zgroup'), 'w') as f:
        json.dump({"zarr_format": 2}, f, indent=2)

def create_pyramid_v2_safe(
    input_path: str,
    output_root: str,
    num_levels: int = 4,
    scale_factors: tuple = (1, 2, 2),
    chunk_size: tuple = (64, 64, 64)
):

    # 1. Define a Context to limit resources
    # This prevents the "bad_alloc" crash by forcing TensorStore to queue tasks
    # rather than running them all at once.
    context_spec = {
        'cache_pool': {'total_bytes_limit': MEMORY_LIMIT_BYTES},
        'data_copy_concurrency': {'limit': CONCURRENCY_LIMIT},
        'file_io_concurrency': {'limit': CONCURRENCY_LIMIT},
    }

    print(f"Opening source: {input_path}")
    source_ts = ts.open({
        'driver': 'zarr', 
        'kvstore': {'driver': 'file', 'path': input_path},
        'context': context_spec  # <--- Apply limits here
    }).result()

    if len(scale_factors) != source_ts.ndim:
        raise ValueError(f"Scale factors {scale_factors} do not match dimensions {source_ts.ndim}")

    dtype = source_ts.dtype.numpy_dtype
    current_source = source_ts

    for level in range(1, num_levels + 1):
        target_path = f"{output_root}/s{level}"

        print(f"--- Processing Level s{level} (Safe Mode) ---")
        print(f"Target: {target_path}")

        if os.path.exists(target_path):
            print("Cleaning up existing directory...")
            shutil.rmtree(target_path)

        # Downsample View
        downsampled_view = ts.downsample(current_source, scale_factors, method='mean')

        if dtype != downsampled_view.dtype.numpy_dtype:
            downsampled_view = downsampled_view.astype(dtype)

        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': target_path},
            'context': context_spec, # <--- Apply limits here as well
            'metadata': {
                'shape': downsampled_view.shape,
                'dtype': dtype.str,
                'chunks': chunk_size,
                'compressor': {
                    'id': 'zstd', 
                    'level': 3, 
                },
                'dimension_separator':'/',
                'fill_value':0,
                'order': 'C',
                'zarr_format': 2
            }
        }

        # Create and Write
        print("Opening target directory...")
        target_ts = ts.open(spec, create=True).result()

        # This write will now obey the concurrency limits set in 'context_spec'
        print("Writing target directory...")
        for z in range(0, downsampled_view.shape[0], chunk_size[0]):
            zs = range(z, min(downsampled_view.shape[0], z+chunk_size[0]))
            if not downsampled_view[zs,...].storage_statistics(query_not_stored=True).result().not_stored:
                print(datetime.now(), 'z =', zs[0], ':', zs[-1], ' saving')
                target_ts[zs,...].write(downsampled_view[zs,...]).result()
            else:
                print(datetime.now(), 'z =', zs[0], ':', zs[-1], ' skipping')

        current_source = target_ts

    write_ome_zarr_v2_metadata(output_root, num_levels, scale_factors)
    print(f"Done. Safe Pyramid located at: {output_root}")

create_pyramid_v2_safe(
    input_path=os.path.join(basepath,inpath),
    output_root=os.path.join(basepath,outpath),
    num_levels=nlevels,
    scale_factors=(2, 2, 2),
    chunk_size=(chunkz, chunkxy, chunkxy) # Your requested chunk size
)
