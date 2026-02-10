using Zarr, NPZ

load_data(basepath, s) = zopen(joinpath(basepath, string("stitched.s",s,".zarr")))

load_invmap(basepath, params) = zopen(joinpath(basepath, string("invmap.",params,".zarr")))

create_warp(shape, chunkxy, chunkz, basepath, params) =
        zcreate(UInt8, shape...;
                path=joinpath(basepath, string("warped.",params,".zarr")),
                chunks=(chunkxy,chunkxy,chunkz),
                fill_value=0,
                compressor=Zarr.ZstdCompressor(level=3),
                dimension_separator='/')

open_warp(basepath, params) = zopen(joinpath(basepath, string("warped.",params,".zarr")), "w")
