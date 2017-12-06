---
title: Variable Selections and Significance of Predictors
notebook: Variable Selections and Significance of Predictors.ipynb
nav_include: 6
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
```


## Load data



```python
df_train = pd.read_csv("data/ADNIMERGE_train.csv")
df_test = pd.read_csv("data/ADNIMERGE_test.csv")
```




```python
X_train = df_train.drop(['RID', 'DX_bl'], axis=1).copy()
y_train = df_train['DX_bl'].copy()
X_test = df_test.drop(['RID', 'DX_bl'], axis=1).copy()
y_test = df_test['DX_bl'].copy()
```


## Significance of Predictors

We would like to find out the most significant variables in the model. There variables have the strongest predicting power, and are thus the most useful in the diagnosis of Alzheimer's disease. Identifying these variables can eliminate the number of tests a patient has to go through to accurately diagnose AD. 

For logistic regression with l1 regularization, we used bootstraping (1000 iterations) to find the most significant predictors. For random forest, we used the returned attribute `feature_importances_`.

### Bootstrap



```python
log_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear')
log_l1.fit(X_train,y_train)
c = log_l1.C_[0]


iterations = 200
boot = np.zeros((X_train.shape[1], iterations))
for i in range(iterations):
    boot_rows = np.random.choice(range(X_train.shape[0]),
                                 size=X_train.shape[0], replace=True)
    X_train_boot = X_train.values[boot_rows]
    y_train_boot = y_train.values[boot_rows]
    model_boot = LogisticRegression(penalty = 'l1', C=c)
    model_boot.fit(X_train_boot, y_train_boot)
    boot[:,i] = model_boot.coef_[2,:]
    
boot_ci_upper = np.percentile(boot, 97.5, axis=1)
boot_ci_lower = np.percentile(boot, 2.5, axis=1)
sig_b_ct = []
for i in range(X_train.shape[1]):
    if boot_ci_upper[i]<0 or boot_ci_lower[i]>0:
        sig_b_ct.append(i)
        
print("Most significant coefficients: ")
print(X_train.columns[sig_b_ct])
```


    Most significant coefficients: 
    Index(['ADAS13', 'MMSE', 'RAVLT_forgetting', 'FAQ'], dtype='object')


Surprisingly, we only have 4 significant predictors using bootstrap method, `ADAS13` and `MMSE` (Mini-Mental State Examination) and `RAVLT_forgetting` (Rey Auditory Verbal Learning Test) are all cognitive assessments which are likely to be free or cheap. 

### Feature importance



```python
rf_best = RandomForestClassifier(n_estimators=32, max_depth=6)
rf_best.fit(X_train, y_train)
imp_features = np.array(X_train.columns)[rf_best.feature_importances_!=0]
print("The most important {} features:".format(len(imp_features)))
print(imp_features)
```


    The most important 71 features:
    ['PTGENDER' 'PTEDUCAT' 'PTRACCAT_Asian' 'PTRACCAT_Black' 'PTRACCAT_White'
     'PTETHCAT_Not_Hisp/Latino' 'PTMARRY_Married' 'PTMARRY_Never_married'
     'PTMARRY_Widowed' 'APOE4' 'CSF_ABETA' 'CSF_TAU' 'CSF_PTAU' 'FDG'
     'FDG_slope' 'AV45' 'AV45_slope' 'ADAS13' 'ADAS13_slope' 'MMSE'
     'MMSE_slope' 'RAVLT_immediate' 'RAVLT_immediate_slope' 'RAVLT_learning'
     'RAVLT_learning_slope' 'RAVLT_forgetting' 'RAVLT_forgetting_slope'
     'RAVLT_perc_forgetting' 'RAVLT_perc_forgetting_slope' 'MOCA' 'MOCA_slope'
     'EcogPtMem' 'EcogPtMem_slope' 'EcogPtLang' 'EcogPtLang_slope'
     'EcogPtVisspat' 'EcogPtVisspat_slope' 'EcogPtPlan' 'EcogPtPlan_slope'
     'EcogPtOrgan' 'EcogPtOrgan_slope' 'EcogPtDivatt' 'EcogPtDivatt_slope'
     'EcogSPMem' 'EcogSPMem_slope' 'EcogSPLang' 'EcogSPLang_slope'
     'EcogSPVisspat' 'EcogSPVisspat_slope' 'EcogSPPlan' 'EcogSPPlan_slope'
     'EcogSPOrgan' 'EcogSPOrgan_slope' 'EcogSPDivatt' 'EcogSPDivatt_slope'
     'FAQ' 'FAQ_slope' 'Ventricles' 'Ventricles_slope' 'Hippocampus'
     'Hippocampus_slope' 'WholeBrain' 'WholeBrain_slope' 'Entorhinal'
     'Entorhinal_slope' 'Fusiform' 'Fusiform_slope' 'MidTemp' 'MidTemp_slope'
     'ICV' 'ICV_slope']


Using random forest classifier, we ended up with 71 important features. If the slope of certain variable is not an important feature, we can at least avoid going through the same test again and again in each visit.

## Forward and Backward Selection



```python
def step_forwards_backwards(direction='forward'):
    
    assert direction in ['forward', 'backward']

    predictors = set(X_train.columns)
    selected_predictors = set() if direction=='forward' else set(predictors)
    
    n = X_train.shape[0]
    best_acc = np.inf
    
    best_accuracy = []
    best_models = []
    
    if direction == 'forward':
        X = X_train[list(selected_predictors)].values
        while (True):
            
            possible_scores = []
            possible_predictors = list(selected_predictors ^ predictors)
            
            if len(possible_predictors) == 0:
                break
                
            for predictor in possible_predictors:
                x_temp = np.concatenate([X, X_train[predictor].values.reshape(-1,1)], axis=1)
                rf = RandomForestClassifier(n_estimators=32, max_depth=6)
                rf.fit(x_temp, y_train)
                scores = rf.score(x_temp, y_train)
                possible_scores.append(scores)
                
            best_predictor_ix = np.argmax(possible_scores)
            best_predictor = possible_predictors[best_predictor_ix]
            
            best_acc = np.max(possible_scores)
            best_accuracy.append(best_acc)
            
            selected_predictors.add(best_predictor)            
            X = np.concatenate([X, X_train[best_predictor].values.reshape(-1,1)], axis=1)
            best_models.append(list(selected_predictors))

    else:

        while (True):
            possible_scores = []
            possible_predictors = list(selected_predictors)

            if len(possible_predictors) == 0:
                break

            for predictor in possible_predictors:
                X = np.concatenate([np.ones(n).reshape(-1,1), 
                                    X_train[list(selected_predictors - set([predictor]))].values], 
                                   axis=1)
                if(X.shape[1] != 0):
                    rf = RandomForestClassifier(n_estimators=32, max_depth=6)
                    rf.fit(X, y_train)
                    scores = rf.score(X, y_train)
                    possible_scores.append(scores)

            best_predictor_ix = np.argmax(possible_scores)
            best_predictor = possible_predictors[best_predictor_ix] 

            best_acc = possible_scores[best_predictor_ix]
            selected_predictors.discard(best_predictor)
            
            best_accuracy.append(best_acc)
            best_models.append(list(selected_predictors))
            
    index_of_best_accuracy = np.argmax(best_accuracy)

    return best_models[index_of_best_accuracy]
```




```python
predictors_forward = step_forwards_backwards(direction='forward')
predictors_backward = step_forwards_backwards(direction='backward')
print("Predictors selected by forward selection (", 
      len(predictors_forward), " predictors): \n", predictors_forward)
print("\n-----------------------------------------\n")
print("Predictors selected by backward selection: (", 
      len(predictors_backward), " predictors): \n", predictors_backward)
```


    Predictors selected by forward selection ( 28  predictors): 
     ['MidTemp_slope', 'Fusiform', 'EcogPtVisspat', 'Ventricles_slope', 'MOCA', 'MMSE_slope', 'FDG', 'APOE4', 'MMSE', 'RAVLT_immediate', 'FAQ', 'FAQ_slope', 'ICV_slope', 'EcogPtLang', 'CSF_TAU', 'FDG_slope', 'WholeBrain_slope', 'EcogSPMem', 'ADAS13', 'EcogPtMem', 'EcogPtPlan', 'RAVLT_learning_slope', 'EcogSPDivatt', 'CSF_ABETA', 'EcogPtOrgan_slope', 'RAVLT_forgetting_slope', 'EcogPtLang_slope', 'RAVLT_perc_forgetting_slope']
    
    -----------------------------------------
    
    Predictors selected by backward selection: ( 64  predictors): 
     ['Fusiform_slope', 'EcogPtVisspat', 'RAVLT_immediate_slope', 'RAVLT_perc_forgetting', 'FDG', 'MMSE', 'RAVLT_immediate', 'FAQ', 'PTETHCAT_Not_Hisp/Latino', 'WholeBrain', 'CSF_TAU', 'FDG_slope', 'EcogSPVisspat_slope', 'WholeBrain_slope', 'ICV', 'Entorhinal_slope', 'PTMARRY_Married', 'EcogSPOrgan', 'CSF_PTAU', 'EcogPtOrgan', 'PTRACCAT_More_than_one', 'RAVLT_learning_slope', 'EcogSPDivatt', 'EcogSPVisspat', 'CSF_ABETA', 'RAVLT_forgetting_slope', 'PTRACCAT_Unknown', 'PTRACCAT_Black', 'EcogPtDivatt', 'RAVLT_perc_forgetting_slope', 'PTEDUCAT', 'PTMARRY_Never_married', 'EcogSPPlan_slope', 'MidTemp_slope', 'AV45_slope', 'EcogSPPlan', 'Ventricles_slope', 'MidTemp', 'MMSE_slope', 'APOE4', 'EcogPtDivatt_slope', 'EcogSPLang_slope', 'FAQ_slope', 'Hippocampus_slope', 'EcogPtLang', 'Hippocampus', 'ICV_slope', 'EcogSPMem', 'ADAS13', 'RAVLT_learning', 'Ventricles', 'EcogPtMem', 'PTRACCAT_White', 'RAVLT_forgetting', 'MOCA_slope', 'EcogPtPlan', 'Entorhinal', 'PTMARRY_Widowed', 'PTRACCAT_Hawaiian/Other_PI', 'AV45', 'EcogPtOrgan_slope', 'EcogPtLang_slope', 'ADAS13_slope', 'PTGENDER']


Backward selection chose far more predictors than forward selection.

We can see that genetic analysis such as `APOE4`, CSF biosamples, neuropsychological tests and MRI are the most important varibles in predicting AD. However, only a few variables we get from each of tests are useful, so we do not need to focus on all the testing results. 

Also, notably, we found that it is necessary to take the cognitive assessments and perform brain scan multiple times to check the progress of cognitive decline and brain atrophy. These are very indicative of AD. For the other categories, one test is sufficient for the diagnosis of AD. Going through the same test multiple times will not increase predictability.
