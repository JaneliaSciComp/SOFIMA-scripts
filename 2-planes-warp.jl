# bsub -Phess -n8 -Is /bin/zsh
# julia -t auto ./2-planes-warp.jl data-hess-2-planes.jl /nrs/hess/data/hess_wafers_60_61/export/zarr_datasets/surface-align/run_20251219_110000/pass03-scale2/s3.patch160.stride8.jl flat-w61_serial_080_to_089-w61_s080_r00-top-face.zarr flat-w61_serial_070_to_079-w61_s079_r00-bot-face.zarr 160 8 1024

using ArgParse, UnPack, StaticArrays, Interpolations, DiskArrays
using Morton, ProgressMeter, Dates, ImageTransformations

s = ArgParseSettings()
@add_arg_table! s begin
    "data_loader"
        help = "Data loader module name, e.g., data-test-2-planes"
        required = true
    "basepath"
        help = "filepath to stitched planes"
        required = true
    "top"
        help = "filename of top of one slab"
        required = true
    "bot"
        help = "filename of bottom of an adjacent slab"
        required = true
    "patch_size"
        arg_type=Int
        help="Side length of (square) patch for processing (in pixels, e.g., 32)"
    "stride"
        arg_type=Int
        help="Distance of adjacent patches (in pixels, e.g., 8)"
    "chunk"
        arg_type=Int
        help="of the zarr output"
end

const args = parse_args(ARGS, s)
for (arg,val) in args
    println("$arg => $val")
end

@unpack chunk = args

const min_z1 = 2  # python is 0 offset
const max_z1 = 2

include(args["data_loader"])

const params = string("patch", args["patch_size"], ".stride", args["stride"], ".top", args["top"])

const invmap = -Float32.(load_invmap(args["basepath"], params))

const top, bot = load_data(args["basepath"], args["top"], args["bot"])

const acs = DiskArrays.approx_chunksize(DiskArrays.eachchunk(top))
const ctop = DiskArrays.cache(top, maxsize=9*acs[1]*acs[2]*sizeof(eltype(top)))

const stride = args["stride"]

const sx = range(0, size(top,1)-1, step=stride)
const sy = range(0, size(top,2)-1, step=stride)

const warped = open_warp((2, size(top)...), 2, args["chunk"], args["basepath"], params)

include("warp.jl")

warped[1,:,:] = bot[:,:]

otop = OffsetArray(reshape(ctop, 1, size(ctop)...), 1, 0, 0)
warp_slab(warped, otop, 2:2)
