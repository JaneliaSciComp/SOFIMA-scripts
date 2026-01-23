import os
import requests
import re
import skimage.io as skio
import numpy as np
import tensorstore as ts

url='http://em-services-1.int.janelia.org:8080/render-ws/v1/owner/cellmap/project/jrc_aphid_salivary_1/stack/v3_acquire'

def load_data(basepath, z, s):
    za = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'stitched.s'+str(s)+'.zarr')},
        'open': True,
        }).result()
    return za[z,:,:].read().result()

def save_flow(flow, min_z, max_z, basepath, params, write_metadata):
    r = requests.get(f"{url}/zValues")
    nz = int(float(r.text[1:-1].split(',')[-1]))
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'flow.'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":[flow.shape[0],nz,*flow.shape[2:]],
            "chunks":[flow.shape[0],1,*flow.shape[2:]],
            "fill_value":0,
            'dtype': '<f8',
            'dimension_separator': '/',
        },
        'create': True,
        'open': True,
        'delete_existing': False,
        'assume_metadata': write_metadata==0,
        }).result()[:,min_z+1:max_z+1,...].write(flow).result()

def load_flow(basepath, params, z0, z1):
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'flow.'+params+'.zarr')},
        'open': True,
        }).result()[:,z0+1:z1+1,...].read().result()

def create_mesh(shape, basepath, params, write_metadata):
    r = requests.get(f"{url}/zValues")
    nz = int(float(r.text[1:-1].split(',')[-1]))
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'mesh.'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":[shape[0],nz,*shape[2:]],
            "chunks":[shape[0],1,*shape[2:]],
            "fill_value":0,
            'dtype': '<f8',
            'dimension_separator': '/',
        },
        'create': True,
        'open': True,
        'delete_existing': False,
        'assume_metadata': write_metadata==0,
        }).result()

def write_mesh_plane(fid, plane, z):
    return fid[:,z:z+1,...].write(plane).result()

def load_mesh(basepath, params, z0, z1):
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'mesh.'+params+'.zarr')},
        'open': True,
        }).result()[:,z0+1:z1+1,...].read().result()

def save_invmap(invmap, min_z, max_z, basepath, params):
    r = requests.get(f"{url}/zValues")
    nz = int(float(r.text[1:-1].split(',')[-1]))
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'invmap.'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":[invmap.shape[0],nz,*invmap.shape[2:]],
            "chunks":[invmap.shape[0],1,*invmap.shape[2:]],
            "fill_value":0,
            'dtype': '<f8',
            'dimension_separator': '/',
        },
        'create': True,
        'open': True,
        'delete_existing': False,
        }).result()[:,min_z+1:max_z+1,...].write(invmap).result()

def load_invmap(basepath, params, z0, z1):
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'invmap.'+params+'.zarr')},
        'open': True,
        }).result()[:,z0:z1+1,...].read().result()

def create_warp(shape, chunkxy, chunkz, basepath, params, write_metadata):
    r = requests.get(f"{url}/zValues")
    nz = int(float(r.text[1:-1].split(',')[-1]))
    return ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(basepath, 'warped.'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":[nz,*shape],
            "chunks":[chunkz,chunkxy,chunkxy],
            "fill_value":0,
            'dtype': '|u1',
            'dimension_separator': '/',
        },
        'create': True,
        'open': True,
        'delete_existing': False,
        'assume_metadata': write_metadata==0,
        }).result()

def write_warp_planes(fid, planes, z0, z1):
    return fid[z0:z1,:,:].write(planes).result()
