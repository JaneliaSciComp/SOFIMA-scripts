using Zarr, NPZ

load_data(basepath, top, bot) = zopen(joinpath(basepath, top)), zopen(joinpath(basepath, bot))

load_mesh(basepath, params) = npzread(joinpath(basepath, string("flow.",params,".npy")))
load_invmap(basepath, params) = npzread(joinpath(basepath, string("invmap.",params,".npy")))

open_warp(shape, chunkxy, chunkz, basepath, params) =
        zcreate(UInt8, shape...;
                path=joinpath(basepath, string("warped.",params,".zarr")),
                chunks=(chunkxy,chunkxy,chunkz),
                fill_value=0,
                compressor=Zarr.ZstdCompressor(level=3),
                dimension_separator='/')
