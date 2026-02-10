using ArgParse, UnPack, DiskArrays, OffsetArrays

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

const params = string("patch", args["patch_size"], ".stride", args["stride"], ".scales", replace(args["scales"], ","=>""), ".k0", args["k0"], ".k", args["k"], ".reps", args["reps"])

const invmap = Float32.(load_invmap(args["basepath"], params))

const curr = load_data(args["basepath"], 0)
const acs = DiskArrays.approx_chunksize(DiskArrays.eachchunk(curr))
const ccurr = DiskArrays.cache(curr, maxsize=9*acs[1]*acs[2]*sizeof(eltype(curr)))

const scales_int = parse.(Int, split(args["scales"],','))
const stride = args["stride"]

const s_min = minimum(scales_int)
const stride_min = stride * (2^s_min)

const sx = range(0, size(curr,1)-1, step=stride_min)
const sy = range(0, size(curr,2)-1, step=stride_min)

const warped = open_warp(args["basepath"], params)

include("warp.jl")

cz = collect(cld(min_z1,chunkz)*chunkz : chunkz : max_z1)
(length(cz)==0 || cz[1]!=min_z1) && pushfirst!(cz, min_z1)
(length(cz)==0 || cz[end]!=max_z1) && push!(cz, max_z1)
for iz in 1:length(cz)-1
    warp_slab(warped, ccurr, cz[iz]:cz[iz+1])
end
