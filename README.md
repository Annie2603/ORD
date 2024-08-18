# ORD

## Structure
- Data folder has 
    - original.csv
    - test.csv
    - imbalanced_noord.csv
    - imbalanced_ord.csv
    - METHODS
        - syn_noord.csv
        - syn_ord.csv

1. Preprocess creates test set and a imbalanced version of the original data 
```
python preprocess.py --dataname DATANAME --testsize TESTSIZE --imbalance_ratio IMB --target TARGET
python preprocess.py --dataname "ADULT" --testsize 4000 --imbalance_ratio 0.02 --target income
```
- Ensure original.csv in data/DATANAME folder 
- Test size consists of TESTSIZE instances equally majority and minority
- imbalance_ratio of 0.02 gives 2 minority instances for every 100 majority instances
- Target column is a binary categorical column in original.csv
- OUTPUT: test.csv, imbalanced_noord.csv

2. Used to convert the binary target to ternary target variable.
```
python detect_overlap.py --dataname DATANAME --target INCOME --threshold THRES
python detect_overlap.py --dataname adult --target income --threshold 0.4
```
- Ensure data folder has DATANAME folder with imbalanced_noord.csv file in it
- Threshold value THRES is ideally 0.3, Vary between 0.15 (more overlap) to 0.45 (less overlap)
- Target column is a binary categorical column in imbalanced_noord.csv
- OUTPUT: imbalanced_ord.csv

- NOTE: target in the output file has column name : cond and it has values 0,1,2
    - C00 clear majority - 0
    - C01 overlap majority - 1
    - C1  minority - 2



#### CREATE SYNTHETIC data and save as noord and ord in the same folder
## give naming conventions - method/noord.csv, method/ord.csv
## Burden of training synthesizer and sampling enough minority offloaded? 

python compute_mle.py --augment true/false --method "TABSYN/DDPM/CTABGAN"
## compare the noord, ord with the test in the folder
## compare the noord, ord but along with real minority for classfier

python compute_synthetic_acc.py --method "TABSYN/DDPM/CTABGAN"
## Random oracle : train on real_org \ Imbalanced, test on noord.csv and ord.csv

notebooks are sufficient for auxillary experiments? 
	- like before and after synthesis
	- remove ord from real
	
	
What about toy data? 
- notebook
- end_to_end ? - problem being synthesizers have different methods to run
```