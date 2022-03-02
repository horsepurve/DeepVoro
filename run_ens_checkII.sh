#!/bin/bash

# run_ens_check.sh   - low-level
# run_ens_checkI.sh  - middle-level
# run_ens_checkII.sh - high-level

function do1size()
{
    # $1: resize
    # $2: $dat
    # $3: $shot
    
    echo "##### job: dataset-"${2}" "${3}"-shot -- resize "${1}" #####"
    
    bash run_ens_checkI.sh $1 '0'   'False' $2 $3 # 8 runs inside
    bash run_ens_checkI.sh $1 '90'  'False' $2 $3
    bash run_ens_checkI.sh $1 '180' 'False' $2 $3
    bash run_ens_checkI.sh $1 '270' 'False' $2 $3

    bash run_ens_checkI.sh $1 '0'   'True' $2 $3 # bug within
    bash run_ens_checkI.sh $1 '90'  'True' $2 $3
    bash run_ens_checkI.sh $1 '180' 'True' $2 $3
    bash run_ens_checkI.sh $1 '270' 'True' $2 $3    
    # 8 * 8 runs total
}

dat=$1 # miniImagenet | CUB | tieredImageNet
shot=$2 # 5 | 1

# bash run_ens_checkI.sh 84 '0' 'False' $dat $shot <- example

do1size 84  $dat $shot 
do1size 90  $dat $shot 
do1size 100 $dat $shot 
do1size 110 $dat $shot 
do1size 120 $dat $shot 
do1size 130 $dat $shot 
do1size 140 $dat $shot 
do1size 150 $dat $shot 
# 8 * 8 * 8 runs total

