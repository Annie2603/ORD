# ORD

The official implementation of the paper  
D'souza, Annie, and Sunita Sarawagi. "Synthetic Tabular Data Generation for Imbalanced Classification: The Surprising Effectiveness of an Overlap Class." arXiv preprint arXiv:2412.15657 (2024)."

## Structure
- Naming conventions
- Data folder has 
    - original.csv
    - test.csv
    - imbalanced_noord.csv
    - imbalanced_ord.csv
    - METHODS
        - syn_noord.csv
        - syn_ord.csv

### Python and package versions
- Code runs fine with Python 3.12 and latest version of all packages - pandas, numpy, scikit-learn etc.
```
pandas                   2.2.2
numpy                    2.0.0
matplotlib               3.9.0
scikit-learn             1.5.0
scipy                    1.13.1
sdmetrics                0.14.1
```

## Preprocess creates test set and a imbalanced version of the original data 
```
python preprocess.py --dataname DATANAME --testsize TESTSIZE --imbalance_ratio IMB --target TARGET
python preprocess.py --dataname adult --testsize 4000 --imbalance_ratio 0.02 --target income
```
- Ensure original.csv in data/DATANAME folder 
- Test size consists of TESTSIZE instances equally majority and minority
- imbalance_ratio of 0.02 gives 2 minority instances for every 100 majority instances
- Target column is a binary categorical column in original.csv
- OUTPUT: test.csv, imbalanced_noord.csv

---
## Used to convert the binary target to ternary target variable.
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


----

## Create SYN data and save as noord and ord in the same folder
- Naming conventions - method/syn_noord.csv, method/syn_ord.csv
- Eg. for ctgan create a folder in data/{DATANAME}/ctgan/ and save syn_noord.csv & syn_ord.csv
---
## Compute Machine Learning Efficacy - Main metric in paper
```
python compute_mle.py --dataname adult --target income --method tabsyn
python compute_mle.py --dataname DATANAME --target TARGET --method METHODNAME
```
- Compare the noord, ord with the test in the data folder

---
## Use a classifier from real balanced (oracle) to estimate synthetic data quality
```
 python experiments/synthetic_acc.py --dataname adult --target income --method tabsyn
 python experiments/synthetic_acc.py --dataname DATANAME --target TARGET --method SYNTHESIZERFOLDER
 ```

```
