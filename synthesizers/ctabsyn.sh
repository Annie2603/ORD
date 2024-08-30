#!/bin/bash
# add bash argument dataname and run
# dataname=$1
# run=$2
# cond=$3
# spec_dataname=$4
# gpu=$5
# bash synthesizers/ctabsyn.sh fintech run1 ord fintech_cond 1

# conda activate tabsyn

for run in "run1 run2"
do
    cd /mnt/nas/swethamagesh/tabsyn_vae_cond/tabsyn
    MYPATH='/mnt/nas/swethamagesh/ORD/data/'$1'/ctabsyn/'$run'/'$3
    # create mypath
    mkdir -p $MYPATH

    # python main.py --dataname $4 --method vae --mode train --gpu $5
    python main.py --dataname $4 --method tabsyn --mode train --gpu $5

    # if $3 is ord do the below else do something else
    if [ $3 == "ord" ]
    then
        for i in 1 2 3 4
        do
            python main.py --dataname $4 --method tabsyn --mode sample --condition_by 0 --n_classes 3 --save_path $MYPATH/c0_$i.csv  
            python main.py --dataname $4 --method tabsyn --mode sample --condition_by 1 --n_classes 3 --save_path $MYPATH/c1_$i.csv 
            python main.py --dataname $4 --method tabsyn --mode sample --condition_by 2 --n_classes 3 --save_path $MYPATH/c2_$i.csv 
        done
    else
        for i in 1 2 3 4 
        do
            python main.py --dataname $4 --method tabsyn --mode sample --condition_by 0 --n_classes 2 --save_path $MYPATH/c0_$i.csv  
            python main.py --dataname $4 --method tabsyn --mode sample --condition_by 1 --n_classes 2 --save_path $MYPATH/c1_$i.csv 
        done
    fi


    cd /mnt/nas/swethamagesh/ORD/
    python synthesizers/process_synthesized.py --dataname $1 --method ctabsyn  --run $run --ord noord
    python synthesizers/process_synthesized.py --dataname $1 --method ctabsyn  --run $run --ord ord
done
