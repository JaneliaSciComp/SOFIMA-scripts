#!/usr/bin/env zsh

# launches dependent cluster jobs for each step needed to align a stack of N planes

# usage: ./N-planes-align.sh <data-loader> <basepath> <min-z> <max-z> <patch-size> <stride> <scales> <k0> <k> <repeat> <batch-size> <chunkxy-size> <chunkz-size>
# e.g. ./N-planes-align.sh "data-aphid-N-planes" /nrs/cellmap/arthurb/aphid/stitch.patch16.stride8.scale1.k00.01.k0.1.crop0-0.margin100 10770 10772 50 5 1,2 0.01 0.1 1 2048 1024 2

data_loader=$1
basepath=$2
minz=$3
maxz=$4
patch_size=$5
stride=$6
scales=$7
k0=$8
k=$9
reps=${10}
batch_size=${11}
chunkxy=${12}
chunkz=${13}

jobid_regex='Job <\([0-9]*\)> '

# flow
mesh_dependency=
for z in $(seq $minz $chunkz $maxz); do
    metadata=$((z==minz))
    maxz2=$(( z+chunkz > maxz ? maxz : z+chunkz ))

    params=minz${z}.maxz${maxz2}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}.reps${reps}

    # n1 for 10 planes, n4 for 100 planes
    bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4 -W 1440)
    logfile=$basepath/flow.${params}.log
    grep -lq Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-flow.py $data_loader $basepath $z $maxz2 $patch_size $stride $scales $k0 $k $reps $batch_size $metadata`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    mesh_dependency=${mesh_dependency}done\($jobid\)'&&'
done

params=minz${minz}.maxz${maxz}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}.reps${reps}

midz=$(( (maxz - minz) / 2 + minz ))

# mesh, 1st half reverse order
invmap_dependency=
bsub_flags=(-Pcellmap -n2 -gpu "num=1" -q gpu_l4 -W 10080)
logfile=$basepath/mesh1.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${mesh_dependency%&&} \
    conda run -n multi-sem --no-capture-output \
    python -u ./N-planes-mesh.py $data_loader $basepath $midz $minz $patch_size $stride $scales $k0 $k $reps 1`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
invmap_dependency=${invmap_dependency}done\($jobid\)'&&'

# mesh, 2nd half forward order
bsub_flags=(-Pcellmap -n2 -gpu "num=1" -q gpu_l4 -W 10080)
logfile=$basepath/mesh2.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${mesh_dependency%&&} \
    conda run -n multi-sem --no-capture-output \
    python -u ./N-planes-mesh.py $data_loader $basepath $midz $maxz $patch_size $stride $scales $k0 $k $reps 0`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
invmap_dependency=${invmap_dependency}done\($jobid\)'&&'

# invmap
nprocs=12
bsub_flags=(-Pcellmap -n$nprocs -W 10080)
logfile=$basepath/invmap.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${invmap_dependency%&&} \
    conda run -n multi-sem --no-capture-output \
    python -u ./N-planes-invmap.py $data_loader $basepath $minz $maxz $patch_size $stride $scales $k0 $k $reps $nprocs`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
warp_dependency=done\($jobid\)

# warp
multiscale_dependency=
for z in $(seq $minz $chunkz $maxz); do
    metadata=$((z==minz))
    maxz2=$(( z+chunkz-1 > maxz ? maxz : z+chunkz-1 ))

    params=minz${z}.maxz${maxz2}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}.reps${reps}

    bsub_flags=(-Pcellmap -n24 -W 1440)
    logfile=$basepath/warp.${params}.log
    grep -lq Successfully $logfile && continue
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $warp_dependency \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-warp.py $data_loader $basepath $z $maxz2 $patch_size $stride $scales $k0 $k $reps $chunkxy $chunkz $metadata`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    multiscale_dependency=${multiscale_dependency}done\($jobid\)'&&'
done

# multiscale
params=patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}.reps${reps}
bsub_flags=(-Pcellmap -n8 -W 1440)
logfile=$basepath/multiscale.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${multiscale_dependency%&&} \
    conda run -n multi-sem --no-capture-output \
    python -u ./multiscale.py $basepath warped.$params.zarr multiscale.$params.zarr 4`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
mv_dependency=\($jobid\)
logfile=$basepath/cp.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $mv_dependency \
    mv $basepath/warped.$params.zarr $basepath/multiscale.$params.zarr/s0`
