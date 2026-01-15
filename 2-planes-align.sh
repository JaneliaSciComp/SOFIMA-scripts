# ./2-planes-align.sh <data-loader> <basepath> <patch-size> <stride> <batch-size> <chunk>

# ./2-planes-align.sh data-hess-2-planes /nrs/hess/data/hess_wafers_60_61/export/zarr_datasets/surface-align/run_20251219_110000/pass03-scale2 flat-w61_serial_080_to_089-w61_s080_r00-top-face.zarr flat-w61_serial_070_to_079-w61_s079_r00-bot-face.zarr 160 8 1024 1024

data_loader=$1
basepath=$2
top=$3
bot=$4
patch_size=$5
stride=$6
batch_size=$7
chunk=$8

params=patch${patch_size}.stride${stride}

jobid_regex='Job <\([0-9]*\)> '

bsub_flags=(-Phess -n4 -gpu "num=1" -q gpu_l4 -W 1440)
logfile=$basepath/flow-mesh-$params.log
bsub_stdout=`bsub ${bsub_flags[@]} -oo $logfile \
    conda run -n multi-sem --no-capture-output \
    python -u ./2-planes-flow-mesh.py $data_loader $basepath $top $bot $patch_size $stride $batch_size`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
dependency=done\($jobid\)

bsub_flags=(-Phess -n32 -W 1440)
logfile=$basepath/invmap-$params.log
bsub_stdout=`bsub ${bsub_flags[@]} -w "$dependency" -oo $logfile \
    conda run -n multi-sem --no-capture-output \
    python -u ./2-planes-invmap.py $data_loader $basepath $top $patch_size $stride`
jobid=`expr match "$bsub_stdout" "$jobid_regex"`
dependency=done\($jobid\)

bsub_flags=(-Phess -n8 -W 1440)
logfile=$basepath/warp-$params.log
bsub_stdout=`bsub ${bsub_flags[@]} -w "$dependency" -oo $logfile \
    conda run -n multi-sem --no-capture-output \
    python -u ./2-planes-warp.py $data_loader $basepath $top $bot $patch_size $stride $chunk`
