#!/usr/bin/env zsh

# launches dependent cluster jobs for each step needed to align a stack of N planes

# usage: ./N-planes-align.sh <data-loader> <basepath> <min-z> <max-z> <patch-size> <stride> <scales> <k0> <k> <repeat> <batch-size> <chunkxy-size> <chunkz-size>
# e.g. ./N-planes-align.sh "data-aphid-N-planes" /nrs/cellmap/arthurb/aphid/stitch.patch16.stride8.scale1.k00.01.k0.1.crop0-0.margin100 10770 10772 50 5 1,2 0.01 0.1 1 2048 1024 2

data_loader=$1
basepath=$2
min_z=$3
max_z=$4
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
mesh_dependency=

for z in $(seq $min_z $chunkz $((max_z-chunkz))); do
    metadata=$((z==min_z))

    params=minz${z}.maxz$((z+chunkz)).patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}.reps${reps}

    bsub_flags=(-Pcellmap -n4 -gpu "num=1" -q gpu_l4)
    logfile=$basepath/flow.${params}.log
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-flow.py $data_loader $basepath $z $((z+chunkz)) $patch_size $stride $scales $k0 $k $reps $batch_size $metadata`
    jobid=`expr match "$bsub_stdout" "$jobid_regex"`
    mesh_dependency=${mesh_dependency}done\($jobid\)'&&'
done

params=minz${min_z}.maxz${max_z}.patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}.reps${reps}

bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4)
logfile=$basepath/mesh.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w ${mesh_dependency%&&} \
    conda run -n multi-sem --no-capture-output \
    python -u ./N-planes-mesh.py $data_loader $basepath $min_z $max_z $patch_size $stride $scales $k0 $k $reps`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
invmap_dependency=done\($jobid\)

bsub_flags=(-Pcellmap -n4)  ### maybe n4, no ???
logfile=$basepath/invmap.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $invmap_dependency \
    conda run -n multi-sem --no-capture-output \
    python -u ./N-planes-invmap.py $data_loader $basepath $min_z $max_z $patch_size $stride $scales $k0 $k $reps`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
warp_dependency=done\($jobid\)

for z in $(seq $min_z $chunkz $((max_z-chunkz))); do
    metadata=$((z==min_z))

    params=minz${z}.maxz$((z+chunkz-1)).patch${patch_size}.stride${stride}.scales${scales//,/}.k0${k0}.k${k}.reps${reps}

    bsub_flags=(-Pcellmap -n8)
    logfile=$basepath/warp.${params}.log
    bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $warp_dependency \
        conda run -n multi-sem --no-capture-output \
        python -u ./N-planes-warp.py $data_loader $basepath $z $((z+chunkz)) $patch_size $stride $scales $k0 $k $reps $chunkxy $chunkz $metadata`
done
