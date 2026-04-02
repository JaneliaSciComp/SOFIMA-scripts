using Zarr, NPZ, HTTP, HDF5, DiskArrays, JSON

const url = "http://em-services-1.int.janelia.org:8080/render-ws/v1/owner/cellmap/project/jrc_velella_b8_1/stack/v3_acquire"

function get_tilepath(Z; url=url)
    r = HTTP.get("$(url)/zRange/$(Z),$(Z)/layoutFile?format=SCHEFFER")
    lines = split(String(r.body), '\n')
    s = Set(replace(line, r"\?.*" => "")[12:end] for line in lines[2:end-1])
    @assert length(s) == 1 "Expected exactly one unique path"
    return first(s)
end

struct VelellaVolume <: AbstractDiskArray{UInt8, 3}
    s::Int
    shape::NTuple{3, Int}
    chunksize::NTuple{3, Int}
end

Base.size(v::VelellaVolume) = v.shape

DiskArrays.haschunks(::VelellaVolume) = DiskArrays.Chunked()

DiskArrays.eachchunk(v::VelellaVolume) =
    DiskArrays.GridChunks(v, v.chunksize)

DiskArrays.approx_chunksize(v::VelellaVolume) = v.chunksize

function DiskArrays.readblock!(v::VelellaVolume, aout, r::AbstractUnitRange...)
    xr, yr, zr = r
    for (iz, z) in enumerate(zr)
        planepath = get_tilepath(z-1)
        h5open(planepath, "r") do fid
            aout[:, :, iz] = fid["0-0-0/mipmap.$(v.s)"][xr, yr, 1]
        end
    end
end

function load_data(basepath, s)
    r = HTTP.get(url)
    meta = JSON.parse(String(r.body))
    zmax = Int(meta["stats"]["stackBounds"]["maxZ"])

    planepath = get_tilepath(zmax)
    nx, ny, chunksize = h5open(planepath, "r") do fid
        ds = fid["0-0-0/mipmap.$s"]
        cx, cy = HDF5.get_chunk(ds)[1:2]
        size(ds, 1), size(ds, 2), (cx, cy, 1)
    end

    VelellaVolume(s, (nx, ny, zmax + 1), chunksize)
end

load_invmap(basepath, params) = zopen(joinpath(basepath, string("invmap.",params,".zarr")))

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
