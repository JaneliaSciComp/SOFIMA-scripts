using OffsetArrays

function warp_chunk(chunk, warped, invmap_xextrema, invmap_yextrema, tform, ccurr, zs)
    out = Array{eltype(warped)}(undef, length(zs), length(chunk[2]), length(chunk[3]))
    for (iz,z) in enumerate(zs)
        ax1 = (max(floor(Int,chunk[2][1]  +invmap_xextrema[iz][1]), 1) :
               min( ceil(Int,chunk[2][end]+invmap_xextrema[iz][2]), size(ccurr,2)))
        ax2 = (max(floor(Int,chunk[3][1]  +invmap_yextrema[iz][1]), 1) :
               min( ceil(Int,chunk[3][end]+invmap_yextrema[iz][2]), size(ccurr,3)))
        oacurr = OffsetArray(ccurr[z, ax1, ax2], ax1, ax2)
        _out = warp(oacurr, tform[iz], (chunk[2],chunk[3]);  method=Lanczos4OpenCV())
        out[iz,:,:] = round.(clamp.(OffsetArrays.no_offset_view(_out),
                                    typemin(eltype(warped)), typemax(eltype(warped))))
    end
    warped[zs,chunk[2],chunk[3]] = out
end

function warp_slab(warped, ccurr, zs)
    invmap_xextrema = [extrema(filter(!isnan, invmap[:,:,z-min_z1+1,1])) for z in zs]
    invmap_yextrema = [extrema(filter(!isnan, invmap[:,:,z-min_z1+1,2])) for z in zs]
    tform = map(zs) do z
        itpx = extrapolate(scale(interpolate(invmap[:,:,z-min_z1+1,1], BSpline(Linear())), sx, sy), NaN32)
        itpy = extrapolate(scale(interpolate(invmap[:,:,z-min_z1+1,2], BSpline(Linear())), sx, sy), NaN32)
        function tform(p::SVector{2})
            x::Float32 = itpx(p[1], p[2])
            y::Float32 = itpy(p[1], p[2])
            SVector{2}(p[1]+x, p[2]+y)
        end
    end

    iz = findfirst(x->all(in.(zs,Ref(x[3]))), DiskArrays.eachchunk(warped)[:,1,1])
    chunks = DiskArrays.eachchunk(warped)[iz,:,:]
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
