#!/usr/bin/env zsh

# launches dependent cluster jobs for each step needed to align a stack of N planes

# usage: ./N-planes-align.sh <data-loader> <basepath> <min-z> <max-z> <scale> <high_pass> <uniform> <threshold> <close> <patch-sizes> <strides> <scales> <k0> <k> <batch-size> <chunkxy-size> <chunkz-size> <num-slices-per-job>
# e.g. ./N-planes-align.sh "data-aphid-N-planes" /nrs/cellmap/arthurb/aphid/stitch.patch16.stride8.scale1.k00.01.k0.1.crop0-0.margin100 10770 10772 2 3 10 2 15 100,50 20,10 1,2 0.01 0.1 2048 1024 2 4

data_loader=$1
basepath=$2
minz=$3
maxz=$4
scale=$5
high_pass=$6
uniform=$7
threshold=$8
close=$9
patch_size=${10}
stride=${11}
scales=${12}
k0=${13}
k=${14}
batch_size=${15}
chunkxy=${16}
chunkz=${17}
nslices=${18}  # integer multiple of chunkz

if (( nslices % chunkz != 0 )) ; then
    echo nslices must be an integer multiple of chunkz
    exit
fi

jobid_regex='Job <\([0-9]*\)> '

# mask
params=minz${minz}.maxz${maxz}.scale${scale}.high_pass${high_pass}.uniform${uniform}.threshold${threshold}.close${close}
bsub_flags=(-Pcellmap -W 10080)
logfile=$basepath/mask.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile \
    julia -e "include(\"${data_loader}.jl\"); \
              curr = load_data(\"$basepath\", 0); \
              params = string(\"scale\", \"$scale\", \".high_pass\", \"$high_pass\", \".uniform\", \"$uniform\", \".threshold\", \"$threshold\", \".close\", \"$close\"); \
              create_mask(size(curr), $chunkxy, \"$basepath\", params)"`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
mask_dependency=done\($jobid\)

flow_dependency=
for z in $(seq $minz $nslices $maxz); do
    metadata=$((z==minz))
    maxz2=$(( z+nslices-1 > maxz ? maxz : z+nslices-1 ))

    params=minz${z}.maxz${maxz2}.scale${scale}.high_pass${high_pass}.uniform${uniform}.threshold${threshold}.close${close}

    # n1 for 10 planes, n4 for 100 planes
    bsub_flags=(-Pcellmap -n1 -W 1440)
    logfile=$basepath/mask.${params}.log
    grep -lqs Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${mask_dependency} \
        julia ./N-planes-mask.jl ${data_loader}.jl $basepath $z $maxz2 $scale $high_pass $uniform $threshold $close`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    flow_dependency=${flow_dependency}done\($jobid\)'&&'
done

# flow
mesh_dependency=
for z in $(seq $minz $nslices $maxz); do
    metadata=$((z==minz))
    maxz2=$(( z+nslices > maxz ? maxz : z+nslices ))

    params=minz${z}.maxz${maxz2}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}

    # n1 for 10 planes, n4 for 100 planes
    bsub_flags=(-Pcellmap -n2 -gpu "num=1" -q gpu_l4 -W 1440)
    logfile=$basepath/flow.${params}.log
    grep -lqs Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${flow_dependency%&&} \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-flow.py $data_loader $basepath $z $maxz2 $scale $high_pass $uniform $threshold $close $patch_size $stride $scales $k0 $k $batch_size $metadata`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    mesh_dependency=${mesh_dependency}done\($jobid\)'&&'
done

# mesh
invmap_dependency=
minz0=$(( minz / nslices * nslices ))  # align slices
for z in $(seq $minz0 $nslices $maxz); do
    minz2=$(( z < minz ? minz : z ))
    maxz2=$(( z+nslices-1 > maxz ? maxz : z+nslices-1 ))
    metadata=$((minz2==minz))

    params=minz${minz2}.maxz${maxz2}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}

    # n1 for 10 planes, n4 for 100 planes
    bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4 -W 10080)
    logfile=$basepath/mesh.${params}.log
    grep -lqs Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${mesh_dependency%&&} \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-mesh.py $data_loader $basepath $minz2 $maxz2 $patch_size $stride $scales $k0 $k $metadata`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    invmap_dependency=${invmap_dependency}done\($jobid\)'&&'
done

# invmap
meshx_dependency=
for z in $(seq $minz $nslices $maxz); do
    metadata=$((z==minz))
    maxz2=$(( z+nslices > maxz ? maxz : z+nslices ))

    params=minz${z}.maxz${maxz2}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}

    bsub_flags=(-Pcellmap -n$nslices -W 10080)
    logfile=$basepath/invmap.${params}.log
    grep -lqs Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${invmap_dependency%&&} \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-invmap.py $data_loader $basepath $z $maxz2 $patch_size $stride $scales $k0 $k $nslices $metadata 0`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    meshx_dependency=${meshx_dependency}done\($jobid\)'&&'
done

params=minz${minz}.maxz${maxz}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}

# meshX
invmapX_dependency=
bsub_flags=(-Pcellmap -n8 -gpu "num=1" -q gpu_l4 -W 10080)
logfile=$basepath/meshX.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${meshx_dependency%&&} \
    conda run -n multi-sem --no-capture-output \
    python -u ./N-planes-meshX.py $data_loader $basepath $minz $maxz $patch_size $stride $scales $k0 $k $nslices`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
invmapX_dependency=${invmapX_dependency}done\($jobid\)'&&'

# invmap
warp_dependency=
for z in $(seq $minz $nslices $maxz); do
    metadata=$((z==minz))
    maxz2=$(( z+nslices > maxz ? maxz : z+nslices ))

    params=minz${z}.maxz${maxz2}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}

    bsub_flags=(-Pcellmap -n$nslices -W 10080)
    logfile=$basepath/invmapX.${params}.log
    grep -lqs Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${invmapX_dependency%&&} \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-invmap.py $data_loader $basepath $z $maxz2 $patch_size $stride $scales $k0 $k $nslices $metadata 1`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    warp_dependency=${warp_dependency}done\($jobid\)'&&'
done

# warp
params=minz${minz}.maxz${maxz}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}
bsub_flags=(-Pcellmap -W 10080)
logfile=$basepath/warp.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w "${warp_dependency%&&}" \
    julia -e "include(\"${data_loader}.jl\"); \
              curr = load_data(\"$basepath\", 0); \
              params = string(\"patch\", \"$patch_size\", \".stride\", \"$stride\", \".scales\", replace(\"$scales\", \",\"=>\"\"), \".k0\", \"$k0\", \".k\", \"$k\"); \
              create_warp(size(curr), $chunkxy, $chunkz, \"$basepath\", params)"`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
warp2_dependency=done\($jobid\)

multiscale_dependency=
minz0=$(( minz / chunkz * chunkz ))
for z in $(seq $minz0 $nslices $maxz); do
    minz2=$(( z < minz ? minz : z ))
    maxz2=$(( z+nslices-1 > maxz ? maxz : z+nslices-1 ))

    params=minz${minz2}.maxz${maxz2}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}

    bsub_flags=(-Pcellmap -n8 -W 1440)
    logfile=$basepath/warp.${params}.log
    grep -lqs Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w "${warp2_dependency%&&}" \
        julia -t auto ./N-planes-warp.jl ${data_loader}.jl $basepath $minz2 $maxz2 $patch_size $stride $scales $k0 $k $chunkxy $chunkz`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    multiscale_dependency=${multiscale_dependency}done\($jobid\)'&&'
done

# multiscale
params=patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}
bsub_flags=(-Pcellmap -n16 -W 1440)
logfile=$basepath/multiscale.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${multiscale_dependency%&&} \
    conda run -n multi-sem --no-capture-output \
    python -u ./multiscale.py $basepath warped.$params.zarr multiscale.$params.zarr 4 $chunkxy $chunkz`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
mv_dependency=\($jobid\)
logfile=$basepath/cp.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $mv_dependency \
    mv $basepath/warped.$params.zarr $basepath/multiscale.$params.zarr/s0`
