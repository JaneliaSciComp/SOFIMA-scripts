# works well with patch_size = 32 and stride = 8

import os
import skimage.io as skio
import numpy as np

zbase = "/groups/scicompsoft/home/preibischs/tavakoli"

def load_data():
    ttop = skio.imread(os.path.join(zbase, "round6_align.tif"))[:,:,49:134]
    tbot = skio.imread(os.path.join(zbase, "round7_align.tif"))[:,:,55:135]
    return ttop, tbot

def crop_data(ttop, tbot):
    nx, ix = ttop.shape[0]//2, ttop.shape[0]//4
    ny, iy = ttop.shape[1]//2, ttop.shape[1]//4
    nz, iz = ttop.shape[2]//2, ttop.shape[2]//4
    ttop_crop = ttop[ix : ix+nx, iy : iy+ny, iz : iz+nz]
    tbot_crop = tbot[ix : ix+nx, iy : iy+ny, iz : iz+nz]
    return ttop_crop, tbot_crop, (iz,iy,ix), (nz,ny,nx)

def tight_crop_data(ttop, tbot):
    nx, ix  = ttop.shape[0]//4, ttop.shape[0]*3//8
    ny, iy  = ttop.shape[1]//4, ttop.shape[1]*3//8
    nz, iz  = ttop.shape[2]//4, ttop.shape[2]*3//8
    ttop_tight_crop = ttop[ix : ix+nx, iy : iy+ny, iz : iz+nz]
    tbot_tight_crop = tbot[ix : ix+nx, iy : iy+ny, iz : iz+nz]
    return ttop_tight_crop, tbot_tight_crop, (iz,iy,ix), (nz,ny,nx)

def save_flow_mesh(flow, mesh, params):
    np.save('1.flow'+params+'.npy', flow)
    np.save('2.mesh'+params+'.npy', mesh)

def load_flow_mesh(params):
    flow = np.load('1.flow'+params+'.npy')
    mesh = np.load('2.mesh'+params+'.npy')
    return flow, mesh

def save_invmap(invmap, params):
    np.save('3.invmap'+params+'.npy', invmap)

def load_invmap(params):
    invmap = np.load('3.invmap'+params+'.npy')
    return invmap
