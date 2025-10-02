#!/usr/bin/env zsh

# launches a cluster job to stitch tiles in a bunch of planes

# usage: ./1-plane-stitch.sh <level> <patch_size> <stride> <k0> <k> <min_z> <max_z> <outpath>
# e.g. ./1-plane-stitch.sh 0 50 5 0.01 0.1 10770 10780 /nrs/cellmap/arthurb/stitch.patch50.stride5.crop30-100.k00.01.k0.1

level=$1
patch_size=$2
stride=$3
k0=$4
k=$5

url='http://em-services-1.int.janelia.org:8080/render-ws/v1/owner/cellmap/project/jrc_aphid_salivary_1/stack/v2_acquire'
MIN_Z=$6
MAX_Z=$7

outpath=$8

bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4)

tiles=$(curl "$url/zRange/${MIN_Z},${MAX_Z}/layoutFile?format=SCHEFFER" | tail -n +2 | cut -d' ' -f2)

uniq_slices=($(echo $tiles | sed 's/^.//' | sed 's/\?.*//' | uniq))
echo nslices = ${#uniq_slices}
for slice in "${uniq_slices[@]}" ; do
    logfile=$outpath/$(basename $slice).log
    bsub ${bsub_flags[@]} -oo $logfile \
            conda run -n multi-sem --no-capture-output \
            python -u ./1-plane-stitch.py "data-aphid-1-plane" $slice $level $patch_size $stride $k0 $k $outpath
done
