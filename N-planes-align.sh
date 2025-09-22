#!/usr/bin/env zsh

# launches dependent cluster jobs for each step needed to align a stack of N planes

# usage: ./N-planes-align.sh <data-loader> <min-z> <max-z> <patch-size> <stride> <batch-size>
# e.g. ./N-planes-align.sh "data-aphid-N-planes" 10770 10780 100 20 256

data_loader=$1
min_z=$2
max_z=$3
patch_size=$4
stride=$5
batch_size=$6

params=minz${min_z}.maxz${max_z}.patch${patch_size}.stride${stride}

jobid_regex='Job <\([0-9]*\)> '

bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4)
logfile=/nrs/cellmap/arthurb/flow.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile conda run -n multi-sem \
    ./N-planes-flow.py $data_loader $min_z $max_z $patch_size $stride $batch_size`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
dependency=done\($jobid\)

bsub_flags=(-Pcellmap -n1)
logfile=/nrs/cellmap/arthurb/mesh.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $dependency conda run -n multi-sem \
    ./N-planes-mesh.py $data_loader $min_z $max_z $patch_size $stride $batch_size`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
dependency=done\($jobid\)

bsub_flags=(-Pcellmap -n1)
logfile=/nrs/cellmap/arthurb/invmap.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $dependency conda run -n multi-sem \
    ./N-planes-invmap.py $data_loader $min_z $max_z $patch_size $stride`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
dependency=done\($jobid\)

bsub_flags=(-Pcellmap -n4)
logfile=/nrs/cellmap/arthurb/warp.${params}.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile -w $dependency conda run -n multi-sem \
    ./N-planes-warp.py $data_loader $min_z $max_z $patch_size $stride`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
dependency=done\($jobid\)
