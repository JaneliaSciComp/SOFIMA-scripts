import os
import re
import skimage.io as skio
import numpy as np
import tensorstore as ts

def load_data(basepath, z, s):
    za = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'stitched.s'+str(s)+'.zarr')},
        'open': True,
        }).result()
    return za[z,:,:].read().result()

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

def open_warp(shape, chunkxy, chunkz, basepath, params):
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'warped.'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":shape,
            "chunks":[chunkz,chunkxy,chunkxy],
            "fill_value":0,
            'dtype': '|u1',
            'dimension_separator': '/',
        },
        'create': True,
        'delete_existing': True,
        }).result()

def write_warp_planes(fid, planes, z0, z1):
    return fid[z0:z1,:,:].write(planes).result()
