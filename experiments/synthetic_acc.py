# # Imports

import sys
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc

import argparse
import os


def get_column_types(df, num_unique_threshold=20):
    """
    This function takes a dataframe and a threshold for the number of unique values
    and returns a dictionary with the numerical and discrete columns.
    """
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    other_cols = [col for col in df.columns if col not in cat_cols]
    other_cols_cat = [col for col in other_cols if df[col].nunique() <= num_unique_threshold]
    discrete_cols = cat_cols + other_cols_cat
    numerical_cols = [col for col in df.columns if col not in discrete_cols]

    return {"numerical_cols": numerical_cols, "discrete_cols": discrete_cols}


# ### Models
# 

def xgboost(X_train, y_train, EPOCHS=200):
    # defining the XGBoost train and test loaders
    xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)

    # defining the hyperparameters and training the model
    n = EPOCHS
    params = {
        'objective': 'binary:logistic',
        
    }

    model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=n)

    # Predicting on the training set
    preds_train = model.predict(xgb_train)
    y_pred_train = [round(pred) for pred in preds_train]
    print(sum(y_pred_train))
    accuracy = accuracy_score(y_train, y_pred_train)
    print('Training Accuracy of the model is:', accuracy*100)
    
    return model

def xgb_predict(model, X_test, y_test, threshold=0.5):
    preds = model.predict(xgb.DMatrix(X_test))
    y_pred = [pred>=threshold for pred in preds]
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy*100)
    preds = np.array([1-preds, preds]).T
    # fpr, tpr, thresholds = roc_curve(y_test, preds[:,1]) 
    # roc_auc = auc(fpr, tpr)
    # print("ROC: ",roc_auc)
    return accuracy*100

# # Train with real data

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--target', type=str, default='income')
parser.add_argument('--method', type=str, default='tabsyn')
parser.add_argument('--run',type=str, default='run1')
args = parser.parse_args()

DATANAME = args.dataname
TARGET = args.target
METHOD = args.method
RUN = args.run

real_data = pd.read_csv(f'data/{DATANAME}/original.csv')

X_train = real_data.drop(TARGET, axis=1)
y_train = real_data[TARGET]
X_test = X_train[:200]
y_test = y_train[:200]

xgboost_model = xgboost(X_train, y_train)
map={}

def acc_no_cond(no_cond, map):
    test0 = no_cond[no_cond[TARGET]==0]
    test1 = no_cond[no_cond[TARGET]==1]

    X_test0 = test0.drop(TARGET, axis=1)
    y_test0 = test0[TARGET]

    X_test1 = test1.drop(TARGET, axis=1)
    y_test1 = test1[TARGET]

    print('C0:', end='')
    map['C0']=xgb_predict(xgboost_model, X_test0, y_test0)
    print('C1:', end='')
    map['C1']=xgb_predict(xgboost_model, X_test1, y_test1)
    return map


def acc_cond(cond,map):
    # apply the condition cond == 2 gets 1 else 0
    cond[TARGET] = cond['cond'].apply(lambda x: 1 if x==2 else 0)
    cond[TARGET].value_counts()
    test0 = cond[cond['cond']==0]
    test1 = cond[cond['cond']==1]
    test2 = cond[cond['cond']==2]

    X_test0 = test0.drop([TARGET, 'cond'], axis=1)
    y_test0 = test0[TARGET]

    X_test1 = test1.drop([TARGET, 'cond'], axis=1)
    y_test1 = test1[TARGET]

    X_test2 = test2.drop([TARGET, 'cond'], axis=1)
    y_test2 = test2[TARGET]

    print('C00: ',end='')
    c0=xgb_predict(xgboost_model, X_test0, y_test0)
    print('C01: ',end='' )
    c1=xgb_predict(xgboost_model, X_test1, y_test1)
    print('C1 (cond):', end='')
    c2=xgb_predict(xgboost_model, X_test2, y_test2)
    map['C00']=c0
    map['C01']=c1
    map['C1(cond)']=c2
    return map


if os.path.exists(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/ord/'):
    files = os.listdir(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/ord/')
    N = len(files)
else:
    print("No files found")
    sys.exit()
if os.path.exists(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/noord/'):
    files = os.listdir(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/noord/')
    N = min(N,len(files))
else:
    print("No files found")
    sys.exit()
N=N-1
no_cond = pd.read_csv(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/noord/set0.csv')
cond = pd.read_csv(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/ord/set0.csv')
for i in range(1,N):
    synth_nc = pd.read_csv(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/noord/set{i}.csv')
    synth_c = pd.read_csv(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/ord/set{i}.csv')
    no_cond=pd.concat([no_cond, synth_nc])
    cond = pd.concat([cond, synth_c])


print('Accuracy of No ord ')
map=acc_no_cond(no_cond,map)
print('Accuracy of ORD')
map=acc_cond(cond,map)

import json
with open(f'data/{DATANAME}/{METHOD}/{RUN}/evaluate/syn_acc.json','w') as file:
    json.dump(map, file)







