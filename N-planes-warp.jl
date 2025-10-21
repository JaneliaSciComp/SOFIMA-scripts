using ArgParse, UnPack, StaticArrays, Interpolations, DiskArrays, OffsetArrays
using Morton, ProgressMeter, Dates, ImageTransformations

s = ArgParseSettings()
@add_arg_table! s begin
    "data_loader"
        help = "Data loader module name, e.g., data-test-2-planes"
        required = true
    "basepath"
        help = "filepath to stitched planes"
        required = true
    "min_z"
        arg_type=Int
        help="lower bound on the planes to align"
    "max_z"
        arg_type=Int
        help="upper bound on the planes to align"
    "patch_size"
        arg_type=Int
        help="Side length of (square) patch for processing (in pixels, e.g., 32)"
    "stride"
        arg_type=Int
        help="Distance of adjacent patches (in pixels, e.g., 8)"
    "scales"
        help="the spatial resolutions to use when computing the flow field"
    "k0"
        arg_type=Float64
        help="spring constant for inter-section springs"
    "k"
        arg_type=Float64
        help="spring constant for intra-section springs"
    "reps"
        arg_type=Int
        help="how many times to iteratively compute the flow"
    "chunkxy"
        arg_type=Int
        help="of the zarr output"
    "chunkz"
        arg_type=Int
        help="of the zarr output"
end

const args = parse_args(ARGS, s)
for (arg,val) in args
    println("$arg => $val")
end

@unpack min_z, max_z, chunkz = args

const min_z1 = min_z + 1  # python is 0 offset
const max_z1 = max_z + 1

include(args["data_loader"])

const params = string("minz", args["min_z"], ".maxz", args["max_z"], ".patch", args["patch_size"], ".stride", args["stride"], ".scales", replace(args["scales"], ","=>""), ".k0", args["k0"], ".k", args["k"], ".reps", args["reps"])

const invmap = Float32.(load_invmap(args["basepath"], params))

const curr = load_data(args["basepath"], 0)
const acs = DiskArrays.approx_chunksize(DiskArrays.eachchunk(curr))
const ccurr = DiskArrays.cache(curr, maxsize=9*acs[1]*acs[2]*sizeof(eltype(curr)))

const stride = args["stride"]

const sx = range(0, size(curr,2)-stride, step=stride)
const sy = range(0, size(curr,1), step=stride)

const warped = open_warp(size(curr), args["chunkxy"], chunkz, args["basepath"], params)

function warp_chunk(chunk, warped, invmap_xextrema, invmap_yextrema, tform, ccurr, zs)
    out = Array{eltype(warped)}(undef, length(chunk[1]), length(chunk[2]), length(zs))
    for (iz,z) in enumerate(zs)
        ax1 = (max(floor(Int,chunk[1][1]  +invmap_xextrema[iz][1]), 1) :
               min( ceil(Int,chunk[1][end]+invmap_xextrema[iz][2]), size(ccurr,1)))
        ax2 = (max(floor(Int,chunk[2][1]  +invmap_yextrema[iz][1]), 1) :
               min( ceil(Int,chunk[2][end]+invmap_yextrema[iz][2]), size(ccurr,2)))
        oacurr = OffsetArray(ccurr[ax1, ax2, z], ax1, ax2)
        _out = warp(oacurr, tform[iz], (chunk[1],chunk[2]);  method=Lanczos4OpenCV())
        out[:,:,iz] = round.(clamp.(OffsetArrays.no_offset_view(_out),
                                          typemin(eltype(warped)), typemax(eltype(warped))))
    end
    warped[chunk[1],chunk[2],zs] = out
end

function warp_slab(warped, ccurr, zs)
    invmap_xextrema = [extrema(filter(!isnan, invmap[1,z-min_z1+1,:,:])) for z in zs]
    invmap_yextrema = [extrema(filter(!isnan, invmap[2,z-min_z1+1,:,:])) for z in zs]
    tform = map(zs) do z
        itpx = extrapolate(scale(interpolate(invmap[1,z-min_z1+1,:,:], BSpline(Linear())), sx, sy), NaN32)
        itpy = extrapolate(scale(interpolate(invmap[2,z-min_z1+1,:,:], BSpline(Linear())), sx, sy), NaN32)
        function tform(p::SVector{2})
            x::Float32 = itpx(p[2], p[1])
            y::Float32 = itpy(p[2], p[1])
            SVector{2}(p[1]+x, p[2]+y)
        end
    end

    iz = findfirst(x->all(in.(zs,Ref(x[3]))), DiskArrays.eachchunk(warped)[1,1,:])
    chunks = DiskArrays.eachchunk(warped)[:,:,iz]
    p = Progress(prod(size(chunks)))
    Threads.@threads :greedy for i in 1:maximum(size(chunks))^2
        ix, iy = morton2cartesian(i)
        (ix > size(chunks,1) || iy > size(chunks,2)) && continue
        chunk = chunks[ix,iy]
        warp_chunk(chunk, warped, invmap_xextrema, invmap_yextrema, tform, ccurr, zs)
        next!(p)
    end
    finish!(p)
end

warped[:,:,min_z1] = curr[:,:,min_z1]

cz = collect(cld(min_z1,chunkz)*chunkz : chunkz : max_z1)
(length(cz)==0 || cz[1]!=min_z1) && pushfirst!(cz, min_z1)
(length(cz)==0 || cz[end]!=max_z1) && push!(cz, max_z1)
for iz in 1:length(cz)-1
    warp_slab(warped, ccurr, cz[iz]+1:cz[iz+1])
end
