import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import argparse

# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV

def boundary_function_helper(X_train, y_train, X_test, y_test, test, THRESHOLD=0.4):
    EPOCHS = 200

    #hyperparemeter tuning
    param_dist = {'n_estimators': [50]}
    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_

    y_pred = best_rf.predict(X_test)
    y_pred_train = best_rf.predict(X_train)
    proba_test = best_rf.predict_proba(X_test)
    proba_train = best_rf.predict_proba(X_train)
    

    uncertain_train = 0
    for prob in proba_train:
        if prob[0] >= THRESHOLD and prob[1] >= THRESHOLD:
            uncertain_train += 1 
    uncertain_test = 0
    for prob in proba_test:
        if prob[0] >= THRESHOLD and prob[1] >= THRESHOLD:
            uncertain_test += 1 
    print(f"{uncertain_train} out of {len(proba_train)} samples have higher uncertainity from the training set")
    print(f"{uncertain_test} out of {len(proba_test)} samples have higher uncertainity from the test set")

    boundary = []
    for i in range(len(y_test)):
        if (y_test[i] != y_pred[i]) or (proba_test[i][0] >= THRESHOLD and proba_test[i][1] >= THRESHOLD):
            boundary.append(i)
    print(f"There are {len(boundary)} points that have been wrongly predicted or are at the boundary")
        # Finding boundary samples from the test set by thresholding on the prediction probabilities or checking if a sample has been incorrectly predicted
    
    # creating a temporary dataframe that contains isBoundary=1 for the samples in the boundary list and 0 for others
    df1 = test
    df1['isBoundary'] = 0
    for wrong in boundary:
        df1.loc[wrong, "isBoundary"] = 1

    return df1

def find_boundary(df, TARGET,  RANDOM_STATE=42, threshold=0.4):
        
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    df_class1 = df[df[TARGET] == 1]
    df_class0 = df[df[TARGET] == 0]
    df = pd.concat([df_class0, df_class1], axis=0)
    
    start = [0, len(df_class0)//2]
    end = [len(df_class0)//2, len(df_class0)]

    for i in range(2):
        print(f"Split {i+1}")
        train = df.drop([k for k in range(start[i], end[i])], axis=0)
        test = df.drop(train.index, axis=0)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        X_train = train.drop(TARGET, axis=1)
        y_train = train[TARGET]
        X_test = test.drop(TARGET, axis=1)
        y_test = test[TARGET]
        
        # get boundary dataframe
        if i == 0:
            df1 = boundary_function_helper(X_train, y_train, X_test, y_test, test, threshold)
        else:
            df2 = boundary_function_helper(X_train, y_train, X_test, y_test, test, threshold)

    
    bnd =  pd.concat([df1, df2], axis=0)
    # print isBoundary=1 and target=0 count
    print("Majority Boundary = ",bnd[(bnd['isBoundary'] == 1) & (bnd[TARGET] == 0)].shape[0])
    print('-'*100)

    bnd['cond'] = 0
    bnd.loc[(bnd['isBoundary'] == 1) & (bnd[TARGET] == 0), 'cond'] = 1
    # remove isBoundary, target column
    bnd = bnd.drop(['isBoundary', TARGET], axis=1)

    # add class 1 as cond 2
    df_class1 = df_class1.copy()
    df_class1.loc[:, 'cond'] = 2
    df_class1 = df_class1.drop(TARGET, axis=1)

    bnd = pd.concat([bnd, df_class1], axis=0)

    return bnd

# get as arguments while running 

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--target', type=str, default='target')
args = parser.parse_args()


DATANAME = args.dataname
THRESHOLD = args.threshold
TARGET = args.target

path = f'data/{DATANAME}/imbalanced_noord.csv'
df = pd.read_csv(path)
bndry = find_boundary(df, TARGET, threshold=THRESHOLD)
bndry.to_csv(f'data/{DATANAME}/imbalanced_ord.csv', index=False)

print("Saved Ternary Target successfully")


