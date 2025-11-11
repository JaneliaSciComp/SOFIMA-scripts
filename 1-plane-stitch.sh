#!/usr/bin/env zsh

# launches a cluster job to stitch tiles in a bunch of planes

# usage: ./1-plane-stitch.sh <scale> <patch_size> <stride> <k0> <k> <min_z> <max_z> <outpath> <chunk-size>
# e.g. ./1-plane-stitch.sh 1 16 8 0.01 0.1 150,50,150,50 10770 10780 /nrs/cellmap/arthurb/aphid/stitch.patch16.stride8.scale1.k00.01.k0.1.crop0-0.margin100 1024

scale=$1
patch_size=$2
stride=$3
k0=$4
k=$5
margins=$6
MIN_Z=$7
MAX_Z=$8
outpath=$9
chunk_size=${10}

bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4)

for z in $(seq $MIN_Z $MAX_Z) ; do
    logfile=$outpath/stitched.$z.s$scale.log
    bsub ${bsub_flags[@]} -oo $logfile \
            conda run -n multi-sem --no-capture-output \
            python -u ./1-plane-stitch.py "data-aphid-1-plane" $z $scale $patch_size $stride $k0 $k $margins $outpath $(( z == MIN_Z )) $chunk_size
done
