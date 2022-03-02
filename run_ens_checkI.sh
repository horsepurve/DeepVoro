#!/bin/bash

# run_ens_check.sh   - low-level
# run_ens_checkI.sh  - middle-level
# run_ens_checkII.sh - high-level

split='XXX' # val | novel | XXX to hold it
dat=$4 # miniImagenet | CUB | tieredImageNet
shot=$5 # 1

resize=$1 # 84
degree=$2 # '0'
fli=$3 # 'False'

logf='log'

rm $logf
touch $logf

python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform beta --beta 0.5  >> $logf

python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform beta --beta 0.75  >> $logf

python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform beta --beta 1.0  >> $logf

echo '~~~ log ~~~'
declare -A biases=( ["miniImagenet"]=0.02 ["CUB"]=0.50 ["tieredImageNet"]=0.10)
if [ $shot == 5 ]
then
    bias_from=0.02
    echo "run 5 shot: bias from "${bias_from}   
else
    bias_from="${biases[$dat]}"
    echo "run 1 shot: bias from "${bias_from}
fi

python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform log --bias $bias_from  >> $logf

var=$(echo "scale=2;$bias_from*2" |bc)
python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform log --bias $var  >> $logf

var=$(echo "scale=2;$bias_from*3" |bc)
python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform log --bias $var  >> $logf

var=$(echo "scale=2;$bias_from*4" |bc)
python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform log --bias $var  >> $logf

var=$(echo "scale=2;$bias_from*5" |bc)
python pipeline/run.py --dataname $dat --n_ways 5 --n_shot $shot \
                       --n_queries 15 --n_runs 2000 \
                       --norm True --method s2m2 --dgr $degree \
                       --re_size $resize --flip $fli --nv $split --save_dist \
                       --transform log --bias $var  >> $logf

echo ${1}" | "${2}" | "${3}
grep final $logf | cut -f 3 -d ' '





