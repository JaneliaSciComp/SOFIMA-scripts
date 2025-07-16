# to run unit tests:
#
# em-alignment1.py data-test 16 8 1
# em-alignment2.py data-test 16 8
# em-alignment3.py data-test 16 8

import os
import numpy as np

shift=8
nx,ny=256,128
nxc,nyc=128,64
nxtc,nytc=64,32
dtype=np.uint8

def load_data():
    tmp = np.random.randint(0, np.iinfo(dtype).max//2, size=(nx+shift,ny+shift), dtype=dtype)
    tmp[nx//2:shift+nx//2, ny//2:shift+ny//2] = np.iinfo(dtype).max
    ttop = tmp[:nx, :ny]
    tbot = tmp[shift:shift+nx, shift:shift+ny]
    return ttop, tbot

def crop_data(ttop, tbot):
    ix = (nx-nxc)//2
    iy = (ny-nyc)//2
    return ttop[ix:ix+nxc, iy:iy+nyc], tbot[ix:ix+nxc, iy:iy+nyc], (iy,ix), (nyc,nxc)

def tight_crop_data(ttop, tbot):
    ix = (nx-nxtc)//2
    iy = (ny-nytc)//2
    return ttop[ix:ix+nxtc, iy:iy+nytc], tbot[ix:ix+nxtc, iy:iy+nytc], (iy,ix), (nytc,nxtc)

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
