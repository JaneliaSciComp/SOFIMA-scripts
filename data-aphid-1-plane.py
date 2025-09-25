import os
import h5py
import numpy as np
import skimage.io as skio

tile_space = (2, 3)

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
                tile_map[(x, y)] = np.array(d[0,:,100:])

    return tile_map

def save_plane(outpath, planepath, stitched, level):
    planename = os.path.basename(planepath)
    skio.imsave(f'{outpath}/{planename}-s{level}.tif', stitched)

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
