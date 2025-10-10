#!/usr/bin/env zsh

# launches a cluster job to stitch tiles in a bunch of planes

# usage: ./1-plane-stitch.sh <level> <patch_size> <stride> <k0> <k> <min_z> <max_z> <outpath>
# e.g. ./1-plane-stitch.sh 0 50 5 0.01 0.1 10770 10780 /nrs/cellmap/arthurb/stitch.patch50.stride5.crop30-100.k00.01.k0.1 1024

level=$1
patch_size=$2
stride=$3
k0=$4
k=$5
MIN_Z=$6
MAX_Z=$7
outpath=$8
chunk_size=$9

bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4)

for z in $(seq $MIN_Z $MAX_Z) ; do
    logfile=$outpath/stitched.$z.s$level.log
    bsub ${bsub_flags[@]} -oo $logfile \
            conda run -n multi-sem --no-capture-output \
            python -u ./1-plane-stitch.py "data-aphid-1-plane" $z $level $patch_size $stride $k0 $k $outpath $(( z == MIN_Z )) $chunk_size
done
