import os
import re
import numpy as np
import tensorstore as ts

def load_data(basepath, top, bot):
    ztop = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, top)},
        'open': True,
        }).result()
    zbot = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, bot)},
        'open': True,
        }).result()

    return ztop.read().result(), zbot.read().result()
    

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
            'dtype': '|u1',
            'dimension_separator': '/',
        },
        'create': True,
        'delete_existing': True,
        }).result()

def write_warp_planes(fid, planes, z0, z1):
    return fid[:,:,z0:z1].write(planes).result()


# google cloud
#
#import tensorstore as ts
#
#dataset = ts.open({
#    'driver': 'n5',
#    'kvstore': {
#        'driver': 'gcs',
#        'bucket': 'janelia-spark-test',
#        'path': 'hess_wafers_60_61_export/render/w61_serial_100_to_109/w61_s109_r00_gc_par_align_ic2d___20251007-104445/s0'
#    },
#    'open': True
#}).result()
#
#dataset.shape
#
#dataset[:100,:100,1].read().result()
