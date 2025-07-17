# to run unit tests:
#
# 3d-flow-mesh.py data-test-3d 32 8 2
# 3d-invmap.py data-test-3d 32 8
# 3d-test.py data-test-3d 32 8

import os
import numpy as np

shift=4
nx,ny,nz=256,128,64
nxc,nyc,nzc=128,64,32
nxtc,nytc,nztc=64,32,16
dtype=np.uint8

def load_data():
    tmp = np.random.randint(0, np.iinfo(dtype).max//2,
                            size=(nx+shift,ny+shift,nz+shift), dtype=dtype)
    tmp[nx//2:shift+nx//2, ny//2:shift+ny//2, nz//2:shift+nz//2] = np.iinfo(dtype).max
    ttop = tmp[:nx, :ny, :nz]
    tbot = tmp[shift:shift+nx, shift:shift+ny, shift:shift+nz]
    return ttop, tbot

def crop_data(ttop, tbot):
    ix = (nx-nxc)//2
    iy = (ny-nyc)//2
    iz = (nz-nzc)//2
    return ttop[ix:ix+nxc, iy:iy+nyc, iz:iz+nzc], tbot[ix:ix+nxc, iy:iy+nyc, iz:iz+nzc], \
           (iz,iy,ix), (nzc,nyc,nxc)

def tight_crop_data(ttop, tbot):
    ix = (nx-nxtc)//2
    iy = (ny-nytc)//2
    iz = (nz-nztc)//2
    return ttop[ix:ix+nxtc, iy:iy+nytc, iz:iz+nztc], tbot[ix:ix+nxtc, iy:iy+nytc, iz:iz+nztc], \
           (iz,iy,ix), (nztc,nytc,nxtc)

def save_flow_mesh(flow, mesh, params):
    np.save('1.flow'+params+'.npy', flow)
    np.save('2.mesh'+params+'.npy', mesh)

def load_flow_mesh(params):
    flow = np.load('1.flow'+params+'.npy')
    mesh = np.load('2.mesh'+params+'.npy')
    return flow, mesh

def save_map(invmap, params):
    np.save('3.invmap'+params+'.npy', invmap)

def load_map(params):
    invmap = np.load('3.invmap'+params+'.npy')
    return invmap
