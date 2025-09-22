#!/usr/bin/env zsh

# launches a cluster job to stitch tiles in a bunch of planes

# usage: ./1-plane-stitch.sh <level> <patch_size> <stride> <min_z> <max_z>

level=$1
patch_size=$2
stride=$3

bsub_flags=(-Pcellmap -n1 -gpu "num=1" -q gpu_l4)

url='http://em-services-1.int.janelia.org:8080/render-ws/v1/owner/cellmap/project/jrc_aphid_salivary_1/stack/v2_acquire'
MIN_Z=$4
MAX_Z=$5

tiles=$(curl "$url/zRange/${MIN_Z},${MAX_Z}/layoutFile?format=SCHEFFER" | tail -n +2 | cut -d' ' -f2)

uniq_slices=($(echo $tiles | sed 's/^.//' | sed 's/\?.*//' | uniq))
echo nslices = ${#uniq_slices}
for slice in "${uniq_slices[@]}" ; do
    logfile=$PWD/aphid/$(basename $slice).log
    bsub ${bsub_flags[@]} -o $logfile conda run -n multi-sem \
                ./1-plane-stitch.py "data-aphid-1-plane" $slice $level $patch_size $stride
done
