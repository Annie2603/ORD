#!/bin/bash
# add bash argument dataname and run
# dataname=$1
# run=$2
# cond=$3
# spec_dataname=$4
# gpu=$5
# bash synthesizers/tabsyn.sh fintech run1 ord fintech_cond 1
# conda activate tabsyn

for run in "run1" 
do

        cd /mnt/nas/swethamagesh/tabsyn-fresh/tabsyn
        MYPATH='/mnt/nas/swethamagesh/ORD/data/'$1'/tabsyn/'$run'/'$3
        # create mypath
        mkdir -p $MYPATH
        python main.py --dataname $4 --method vae --mode train --gpu $5
        python main.py --dataname $4 --method tabsyn --mode train --gpu $5
        # loop over the below three lines n times
        for i in 1 2 3  
        do
            python main.py --dataname $4 --method tabsyn --mode sample --save_path $MYPATH/one$i.csv 
            python main.py --dataname $4 --method tabsyn --mode sample --save_path $MYPATH/two$i.csv 
            python main.py --dataname $4 --method tabsyn --mode sample --save_path $MYPATH/three$i.csv 
            python main.py --dataname $4 --method tabsyn --mode sample --save_path $MYPATH/four$i.csv 
            python main.py --dataname $4 --method tabsyn --mode sample --save_path $MYPATH/five$i.csv 
            python main.py --dataname $4 --method tabsyn --mode sample --save_path $MYPATH/six$i.csv 
        done

        cd /mnt/nas/swethamagesh/ORD/
        # create folders in the data directory
        python synthesizers/process_synthesized.py --dataname $1 --method tabsyn  --run $run --ord ord

done