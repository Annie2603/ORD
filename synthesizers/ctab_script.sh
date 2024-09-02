# bash synthesizers/ctabsyn.sh fintech run1 ord fintech_cond 1 > logs/ctab_fin.log
# bash synthesizers/ctabsyn.sh fintech run1 noord fintech 1  > log/ctab_fin1.log
bash synthesizers/ctabsyn.sh heloc run1 ord heloc_cond 1 > logs/ctab_helio.log
bash synthesizers/ctabsyn.sh heloc run1 noord heloc 1 > logs/ctab_helio1.log
# bash synthesizers/ctabsyn.sh adult run1 ord adult_cond 1 > logs/ctab_adult.log
# bash synthesizers/ctabsyn.sh adult run1 noord adult_nocond 1  > log/ctab_adult1.log
bash synthesizers/ctabsyn.sh cardio run1 ord cardio_cond 1 > logs/ctab_cardio.log
bash synthesizers/ctabsyn.sh cardio run1 noord cardio 1 > logs/ctab_cardio1.log



# python compute_mle.py --dataname fintech --target churn --method tabsyn --run run1 > logs/fintech_tabsyn_run1.log

