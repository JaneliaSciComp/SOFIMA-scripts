import os
import requests
import re
import h5py
import numpy as np
import skimage.io as skio
import tensorstore as ts

url='http://em-services-1.int.janelia.org:8080/render-ws/v1/owner/cellmap/project/jrc_aphid_salivary_1/stack/v2_acquire'
tile_space = (2, 3)
crop = (slice(30,None), slice(100,None))

def get_tilepath(Z):
    r = requests.get(f"{url}/zRange/{Z},{Z}/layoutFile?format=SCHEFFER")
    s = set([re.sub(r"\?.*","", s)[11:] for s in r.text.split('\n')[1:-1]])
    assert len(s)==1
    return s.pop()

def load_data(planepath, level):
    # Define the tile space. This specifies how the different tiles are distributed
    # in space, and should normally be derived from the metadata provided by the
    # microscope.
    tile_id_map = [["0-0-0", "0-0-1", "0-0-2"],
                   ["0-1-0", "0-1-1", "0-1-2"]]
    tile_id_map = np.array(tile_id_map)

    # Load tile images.
    tile_map = {}

    for y in range(tile_id_map.shape[0]):
        for x in range(tile_id_map.shape[1]):
            tile_id = tile_id_map[y, x]
            with h5py.File(f'{planepath}', 'r') as fid:
                d = fid[f'{tile_id}/mipmap.{level}']
                tile_map[(x, y)] = np.array(d[0,*crop])

    return tile_map

def save_plane(outpath, z, stitched, level, write_metadata, chunk_size):
    r = requests.get(f"{url}/zValues")
    nz = int(float(r.text[1:-1].split(',')[-1]))
    za = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(outpath, 'stitched.s'+str(level)+'.zarr')},
        'metadata': {
            "compressor": {"id":"zstd", "level":3},
            "shape": [nz,*stitched.shape],
            "chunks": [1,chunk_size,chunk_size],
            "fill_value": 0,
            'dtype': '|u1',
            'dimension_separator': '/',
        },
        'create': True,
        'open': True,
        'delete_existing': False,
        'assume_metadata': write_metadata==1,
        }).result()
    za[z,:,:] = stitched

def save_tiles(outpath, planepath, stitched, maxdim):
    planename = os.path.basename(planepath)
    padded_array = np.zeros(maxdim, dtype=np.uint8)
    for k in stitched.keys():
        padded_array[:,:] = 0
        sh = stitched[k].shape
        padded_array[:sh[0],:sh[1]] = stitched[k]
        x = list(k)
        x.reverse()
        x = [0]+x
        skio.imsave(f'{outpath}/{planename}-{x}.tif'.replace(" ",""), padded_array)
