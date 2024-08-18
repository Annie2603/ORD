'''
```
python preprocess.py --dataname DATANAME --testsize TESTSIZE --imbalance_ratio IMB --target TARGET
python preprocess.py --dataname adult --testsize 4000 --imbalance_ratio 0.02 --target income
```
- Ensure original.csv in data/DATANAME folder 
- Test size consists of TESTSIZE instances equally majority and minority
- imbalance_ratio of 0.02 gives 2 minority instances for every 100 majority instances
- Target column is a binary categorical column in original.csv

'''
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--testsize', type=int, default=4000)
parser.add_argument('--imbalance_ratio', type=float, default=0.02)
parser.add_argument('--target', type=str, default='income')
args = parser.parse_args()

DATANAME = args.dataname
TESTSIZE = args.testsize
IMB = args.imbalance_ratio
TARGET = args.target

TEST_CLASS = TESTSIZE//2


real = pd.read_csv(f'data/{DATANAME}/original.csv')

# choose 2000 samples from the majority class
test0 = real[real[TARGET]==0].sample(TEST_CLASS, random_state=42)
test1 = real[real[TARGET]==1].sample(TEST_CLASS, random_state=42)


test = pd.concat([test0, test1])
remaining = real[~real.index.isin(test.index)]
test[TARGET].value_counts()

imb0 = remaining[remaining[TARGET]==0]
min_count = int(IMB * len(imb0))

imb1 = remaining[remaining[TARGET]==1].sample(min_count, random_state=42)
imb_df = pd.concat([imb1, imb0])
# shuffle
imb_df = imb_df.sample(frac=1, random_state=42).reset_index(drop=True)
test = test.sample(frac=1, random_state=42).reset_index(drop=True)

imb_df.to_csv(f'data/{DATANAME}/imbalanced_noord.csv', index=False)
test.to_csv(f'data/{DATANAME}/test.csv', index=False)


