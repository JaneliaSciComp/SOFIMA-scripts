using Zarr, NPZ

load_data(basepath, s) = zopen(joinpath(basepath, string("stitched.s",s,".zarr")))

load_invmap(basepath, params) = zopen(joinpath(basepath, string("invmapX.",params,".zarr")))

create_mask(shape, chunkxy, basepath, params) =
        zcreate(UInt8, shape...;
                path=joinpath(basepath, string("mask.",params,".zarr")),
                chunks=(chunkxy,chunkxy,1),
                fill_value=false,
                compressor=Zarr.ZstdCompressor(level=3),
                dimension_separator='/')

open_mask(basepath, params) = zopen(joinpath(basepath, string("mask.",params,".zarr")), "w")

create_warp(shape, chunkxy, chunkz, basepath, params) =
        zcreate(UInt8, shape...;
                path=joinpath(basepath, string("warped.",params,".zarr")),
                chunks=(chunkxy,chunkxy,chunkz),
                fill_value=0,
                compressor=Zarr.ZstdCompressor(level=3),
                dimension_separator='/')

open_warp(basepath, params) = zopen(joinpath(basepath, string("warped.",params,".zarr")), "w")
