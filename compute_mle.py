
import sys
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import CalibratedClassifierCV

import sdmetrics
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import (BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression, BinaryMLPClassifier)
import argparse



def expected_calibration_error(samples, true_labels, M=10, threshold=0.5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)
    predicted_label = (np.array([x[1] for x in samples]) >= threshold).astype(int)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels
    # print fraction of true
    print(f"Fraction of true: {accuracies.mean()}")
    # print(accuracies)

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()
        # round each val to 3 decimal places

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()

            # print(f"Bin: {bin_lower.item():.3f} - {bin_upper.item():.3f} | Frac: {prob_in_bin:.3f} | Accuracy: {accuracy_in_bin:.3f}")
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


def xgboost_train(X_train, y_train, EPOCHS=200):
    # defining the XGBoost train and test loaders
    xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)

    # defining the hyperparameters and training the model
    n = EPOCHS
    params = {
        'objective': 'binary:logistic',
    }

    model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=n)   
    return model

def xgb_predict(model, X_test, y_test, threshold=0.5):
    preds = model.predict(xgb.DMatrix(X_test))
    y_pred = [pred>=threshold for pred in preds]
    print(sum(y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print('XG: ', accuracy*100)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    # find minority class accuracy 
    print("Minority class accuracy: ", recall_score(y_test, y_pred, pos_label=1)*100)
    print("Majority class accuracy: ", recall_score(y_test, y_pred, pos_label=0)*100)
    # majority class accuracy
    
    preds = np.array([1-preds, preds]).T
    print("ECE ", expected_calibration_error(preds, y_test))
    roc_curve_plot(y_test, preds[:,1]) 
    # return preds, y_pred


def xgboost(X_train, y_train, X_test, y_test, EPOCHS=200):
    model = xgboost_train(X_train, y_train, EPOCHS)
    xgb_predict(model, X_test, y_test)
    return model


def roc_curve_plot(y_test, y_pred_proba):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Classification')
    plt.legend()
    plt.show()

def find_meta_data(df, UNIQ_THRESHOLD=20):
    """segregating all columns into categorical (contains object type, integer
     and float type with lesser than UNIQ_THRESHOLD unique elements), integer type and float type"""

    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    int_cols = [col for col in df.columns if df[col].dtype == 'int64']
    float_cols = [col for col in df.columns if df[col].dtype == 'float64']
    disc_int_cols = [col for col in int_cols if df[col].nunique()
                     < UNIQ_THRESHOLD]
    disc_float_cols = [
        col for col in float_cols if df[col].nunique() < UNIQ_THRESHOLD]
    discrete_cols = cat_cols + disc_int_cols + disc_float_cols
    int_cols1 = [col for col in int_cols if col not in disc_int_cols]
    float_cols1 = [col for col in float_cols if col not in disc_float_cols]


    # Defining the metadata as is required by SDMetrics
    conti_cols = int_cols1 + float_cols1 
    metadata = dict()
    column_dict = dict()
    for col in conti_cols:
        column_dict[col] = {"sdtype": "numerical"}
    for col in discrete_cols:
        column_dict[col] = {"sdtype": "categorical"}
    metadata['columns']  = column_dict
    return metadata


def get_report(real_df, syn_df, metadata):
    report = QualityReport()
    report.generate(real_df, syn_df, metadata)
    print(report)
    return report

def MachineLearningAccuracy(test, train, metadata):
    ada = BinaryAdaBoostClassifier.compute(
    test_data=test,
    train_data=train,
    target=TARGET,
    metadata=metadata
    )
    dt = BinaryDecisionTreeClassifier.compute(
        test_data=test,
        train_data=train,
        target=TARGET,
        metadata=metadata
    )
    lr = BinaryLogisticRegression.compute(
        test_data=test,
        train_data=train,
        target=TARGET,
        metadata=metadata
    )

    mlp = BinaryMLPClassifier.compute(
        test_data=test,
        train_data=train,
        target=TARGET,
        metadata=metadata
    )
    print(f"Scores - \nada: {ada}, \ndt: {dt}, \nlr: {lr}, \nmlp: {mlp}")
    print(f"avg: {(ada+dt+lr+mlp)/4}")
    return (ada+dt+lr+mlp)/4
    

DATANAME = 'adult'
TARGET = 'income'
METHOD = 'tabsyn'
# parameters argument for the 3 using argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--target', type=str, default='income')
parser.add_argument('--method', type=str, default='tabsyn')
args = parser.parse_args()

DATANAME = args.dataname
TARGET = args.target
METHOD = args.method



test = pd.read_csv(f'data/{DATANAME}/test.csv')
X_test = test.drop(TARGET, axis=1)
y_test = test[TARGET]


# # MLE - cond02 classifier


synth = pd.read_csv(f'data/{DATANAME}/{METHOD}/syn_ord.csv')

synth[TARGET]= synth['cond'].apply(lambda x: 1 if x==2 else 0)

synth2 = synth[synth['cond']==2]
synth0 = synth[synth['cond']==0].sample(len(synth2), random_state=4)
train = pd.concat([synth2, synth0])
train.drop('cond', axis=1, inplace=True)

X_train = train.drop(TARGET, axis=1)
y_train = train[TARGET]
    
m = xgboost(X_train, y_train, X_test, y_test)


metadata = find_meta_data(train)
MachineLearningAccuracy(test, train, metadata)

# # MLE no cond

synth = pd.read_csv(f'data/{DATANAME}/{METHOD}/syn_noord.csv')

synth2 = synth[synth[TARGET]==1]
synth0 = synth[synth[TARGET]==0].sample(len(synth2), random_state=42)
train = pd.concat([synth2, synth0])

X_train = train.drop(TARGET, axis=1)
y_train = train[TARGET]

m = xgboost(X_train, y_train, X_test, y_test)


metadata = find_meta_data(train)
MachineLearningAccuracy(test, train, metadata)



