import os
import tensorstore as ts

zbase = "/nrs/hess/data/hess_wafers_60_61/export/hess_wafers_60_61.n5/render/w60_serial_360_to_369/w60_s360_r00_d20_gc_align_b_ic/s0"

itop, ibot = 15, 16
icrop_start = (19000,19000)
icrop_size = (2000,2000)
itight_crop_start = (19900,19900)
itight_crop_size = (100,100)

def load_data():
    t = ts.open({
        'driver': 'n5',
        'kvstore': {"driver":"file", "path":zbase},
        }).result()
    _t = t[:30000,:30000,itop:ibot+1].read().result()
    ttop = _t[:,:,0]
    tbot = _t[:,:,-1]
    return ttop, tbot

def crop_data(ttop, tbot):
    ttop_crop = ttop[icrop_start[0] : icrop_start[0]+icrop_size[0],
                     icrop_start[1] : icrop_start[1]+icrop_size[1]]
    tbot_crop = tbot[icrop_start[0] : icrop_start[0]+icrop_size[0],
                     icrop_start[1] : icrop_start[1]+icrop_size[1]]
    return ttop_crop, tbot_crop, icrop_start, icrop_size

def tight_crop_data(ttop, tbot):
    ttop_tight_crop = ttop[itight_crop_start[0] : itight_crop_start[0]+itight_crop_size[0],
                           itight_crop_start[1] : itight_crop_start[1]+itight_crop_size[1]]
    tbot_tight_crop = tbot[itight_crop_start[0] : itight_crop_start[0]+itight_crop_size[0],
                           itight_crop_start[1] : itight_crop_start[1]+itight_crop_size[1]]
    return ttop_tight_crop, tbot_tight_crop, itight_crop_start, itight_crop_size

def save_flow_mesh(flow, mesh, params):
    flow_zarr = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(zbase, '1.flow'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":flow.shape,
            "fill_value":0,
            'dtype': '<f4',
        },
        'create': True,
        'delete_existing': True,
        }).result()
    flow_zarr.write(flow).result()

    solved_zarr = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(zbase, '2.mesh'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":mesh.shape,
            "fill_value":0,
            'dtype': '<f4',
        },
        'create': True,
        'delete_existing': True,
        }).result()
    solved_zarr.write(mesh).result()

def load_flow_mesh(params):
    flow = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(zbase, '1.flow'+params+'.zarr')},
        }).result()
    mesh = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(zbase, '2.mesh'+params+'.zarr')},
        }).result()
    return flow, mesh

def save_map(invmap, params):
    invmap_zarr = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(zbase, '3.invmap'+params+'.zarr')},
        'metadata': {
            "compressor":{"id":"zstd","level":3},
            "shape":invmap.shape,
            "fill_value":0,
            'dtype': '<f8',
        },
        'create': True,
        'delete_existing': True,
        }).result()
    invmap_zarr.write(invmap).result()

def load_map(params):
    invmap = ts.open({
        'driver': 'zarr',
        'kvstore': {"driver":"file", "path":os.path.join(zbase, '3.invmap'+params+'.zarr')},
        }).result()
    invmap = invmap.read().result()
    return invmap
