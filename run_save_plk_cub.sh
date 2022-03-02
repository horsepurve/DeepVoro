#!/bin/bash

function do1size()
{
    local Data='CUB'
    local EPO=255
    local ID=0
    local SIZ=$1

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '0' \
                                                 --re_size $SIZ --flip False

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '90' \
                                                 --re_size $SIZ --flip False

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '180' \
                                                 --re_size $SIZ --flip False

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '270' \
                                                 --re_size $SIZ --flip False

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '0' \
                                                 --re_size $SIZ --flip True

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '90' \
                                                 --re_size $SIZ --flip True

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '180' \
                                                 --re_size $SIZ --flip True

    CUDA_VISIBLE_DEVICES=$ID, python save_plk.py --dataset $Data --method S2M2_R \
                                                 --fetch_epoch $EPO --bvn bvn --dgr '270' \
                                                 --re_size $SIZ --flip True
}

do1size 84
do1size 90
do1size 100
do1size 110
do1size 120
do1size 130
do1size 140
do1size 150
