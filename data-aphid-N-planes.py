import os
import requests
import re
import skimage.io as skio
import numpy as np
import tensorstore as ts

url='http://em-services-1.int.janelia.org:8080/render-ws/v1/owner/cellmap/project/jrc_aphid_salivary_1/stack/v2_acquire'

def get_tile_list(MIN_Z, MAX_Z):
    r = requests.get(f"{url}/zRange/{MIN_Z},{MAX_Z}/layoutFile?format=SCHEFFER")
    return list(dict.fromkeys([os.path.basename(re.sub(r"\?.*","", s)[11:]) for s in r.text.split('\n')[1:-1]]))  # dict.fromkeys() preserves order unlike set()

def load_data(basepath, filenames_noext, z, s):
    return skio.imread(f'{basepath}/{filenames_noext[z]}-s{s}.tif')

def save_flow(flow, basepath, params):
    np.save(os.path.join(basepath, 'flow.'+params+'.npy'), flow)

def load_flow(basepath, params):
    return np.load(os.path.join(basepath, 'flow.'+params+'.npy'))

def save_mesh(mesh, basepath, params):
    np.save(os.path.join(basepath, 'mesh.'+params+'.npy'), mesh)

def load_mesh(basepath, params):
    return np.load(os.path.join(basepath, 'mesh.'+params+'.npy'))

def save_invmap(invmap, basepath, params):
    np.save(os.path.join(basepath, 'invmap.'+params+'.npy'), invmap)

def load_invmap(basepath, params):
    invmap = np.load(os.path.join(basepath, 'invmap.'+params+'.npy'))
    return invmap

def open_warp(shape, chunk_size, basepath, params):
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'warped.'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":shape,
            "chunks":[chunk_size,chunk_size,chunk_size],
            "fill_value":0,
            'dtype': '<u1',
        },
        'create': True,
        'delete_existing': True,
        }).result()

def write_warp_planes(fid, planes, z0, z1):
    return fid[z0:z1,:,:].write(planes)
