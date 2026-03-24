using ArgParse, UnPack, Zarr, Images, Dates

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
    "scale"
        arg_type=Int
        default=2
        help=""
    "high_pass"
        arg_type=Int
        default=3
        help=""
    "uniform"
        arg_type=Int
        default=10
        help=""
    "threshold"
        arg_type=Int
        default=2
        help=""
    "close"
        arg_type=Int
        default=15
        help=""
end

const args = parse_args(ARGS, s)
for (arg,val) in args
    println("$arg => $val")
end

@unpack uniform, min_z, max_z = args

const params = string("scale", args["scale"], ".high_pass", args["high_pass"], ".uniform", uniform, ".threshold", args["threshold"], ".close", args["close"])

include(args["data_loader"])

const s0 = load_data(args["basepath"], 0)
const masked = open_mask(args["basepath"], params)

function doit(sn, scale, high_pass, kernel, threshold, close)
    # downsample
    for _ in 1:scale
        sn = restrict(sn)
    end

    # high-pass filter
    sn .-= imfilter(sn, Kernel.gaussian(high_pass))

    # uniform filter
    uf = imfilter(abs.(sn), kernel, "reflect")

    # threshold
    map!(x -> x .> threshold, uf)

    # close
    cl = closing(uf, strel_diamond(uf, r=close))

    # upsample
    imresize(cl, size(s0)[1:2]) .< 0.5
end

for z in min_z : max_z
    println(now(), " z = ", z);  flush(stdout)
    kernel = centered(ones((uniform,uniform)) ./ uniform^2)
    masked[:,:,z] = doit(s0[:,:,z], args["scale"], args["high_pass"], kernel, args["threshold"], args["close"])
end
