# ./2-planes-align.sh <data-loader> <basepath> <patch-size> <stride> <batch-size>

data_loader=$1
basepath=$2
patch_size=$3
stride=$4
batch_size=$5

params=patch${patch_size}.stride${stride}

jobid_regex='Job <\([0-9]*\)> '

# -n9 needed for 160,32,64 and 150k^2 data
logfile=$basepath/2-planes-flow-mesh-$params.log
cmd="conda run -n multisem python ./2-planes-flow-mesh.py $data_loader $basepath $patch_size $stride $batch_size"
out=`echo $cmd | bsub -Phess -n12 -gpu "num=1" -q gpu_a100 -W 1440 -o $logfile`
jobid=`expr match "$out" "$jobid_regex"`
dep=done\($jobid\)

logfile=$basepath/2-planes-invmap-$params.log
cmd="conda run -n multisem python ./2-planes-invmap.py $data_loader $basepath $patch_size $stride"
echo $cmd | bsub -Phess -n32 -W 1440 -w "$dep" -o $logfile
