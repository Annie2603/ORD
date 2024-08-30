# for a given dataset, collect all the minority and slightly greater majority into synth_ord{n}.csv

import argparse
import pandas as pd
import numpy as np
import json
import os

parser = argparse.ArgumentParser(description='Generation')
parser.add_argument('--dataname', type=str, default='data1', help='name of the dataset')
parser.add_argument('--method', type=str, default='data/', help='path to the dataset')
parser.add_argument('--ord', type=str, default='ord', help='whether it is ord or noord')
parser.add_argument('--run', type=str, default='run1', help='run id')


args = parser.parse_args()

DATANAME = args.dataname
METHOD = args.method
ORD = args.ord
RUN = args.run



PATH = f'data/{DATANAME}/{METHOD}/{RUN}/{ORD}/'
SAVE_PATH = f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/{ORD}/'

# load json 'data.json' 
with open('synthesizers/data.json') as json_file:
    nums = json.load(json_file)
n_data = nums[DATANAME]['syn']
target = nums[DATANAME]['target']

# load data from all csv files in path folder (we don't know how many files there are)
print(SAVE_PATH)
print(ORD)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# delete any files in save path
for file in os.listdir(SAVE_PATH):
    os.remove(SAVE_PATH+file)

data = []
for file in os.listdir(PATH):
    if file.endswith('.csv'):
        data.append(pd.read_csv(PATH + file))
finaldata = pd.concat(data)

if ORD=='ord':
    # split data into k splits of n_data and save as csv
    k = len(finaldata[finaldata['cond']==2])//n_data
    cond1_size = len(finaldata[finaldata['cond']==1])//k
    # cond2_size = len(finaldata[finaldata['cond']==2])//k
    cond2_size = n_data
    cond0_size = cond2_size
    cond1_df = finaldata[finaldata['cond']==1]
    cond2_df = finaldata[finaldata['cond']==2]
    cond0_df = finaldata[finaldata['cond']==0]
    splits = []
    for i in range(k):
        cond1 = cond1_df.iloc[i*cond1_size:(i+1)*cond1_size]
        cond2 = cond2_df.iloc[i*cond2_size:(i+1)*cond2_size]
        cond0 = cond0_df.iloc[i*cond0_size:(i+1)*cond0_size]
        write_df = pd.concat([cond0, cond1, cond2])
        # shuffle
        write_df = write_df.sample(frac=1)
        write_df.to_csv(SAVE_PATH + f'set{i}.csv', index=False)

else:
    k = len(finaldata[finaldata[target]==1])//n_data
    # target1_size = len(finaldata[finaldata[target]==1])//k
    target1_size = n_data
    target0_size = target1_size
    target1_df = finaldata[finaldata[target]==1]
    target0_df = finaldata[finaldata[target]==0]
    # splits without repetition of data into an array of size k
    splits = []
    for i in range(k):
        target1 = target1_df.iloc[i*target1_size:(i+1)*target1_size]
        target0 = target0_df.iloc[i*target0_size:(i+1)*target0_size]
        write_df = pd.concat([target0, target1])
        write_df = write_df.sample(frac=1)
        write_df.to_csv(SAVE_PATH + f'set{i}.csv', index=False)

    

    







