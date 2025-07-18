# ./em-alignment.sh <data-loader> <patch-size> <stride> <batch-size>

basepath=/groups/scicompsoft/home/arthurb/projects/multi-sem
logfile=$basepath/em-alignment-$1-$2-$3-$4.log
jobid_regex='Job <\([0-9]*\)> '

# -n9 needed for 160,32,64 and 150k^2 data
cmd="conda run -n multi-sem python $basepath/em-alignment1.py $1 $2 $3 $4"
out=`echo $cmd | bsub -Psema -n12 -gpu "num=1" -q gpu_a100 -W 1440 -o $logfile`
jobid=`expr match "$out" "$jobid_regex"`
dep=done\($jobid\)

cmd="conda run -n multi-sem python $basepath/em-alignment2.py $1 $2 $3"
echo $cmd | bsub -Psema -n32 -W 1440 -w "$dep" -o $logfile
