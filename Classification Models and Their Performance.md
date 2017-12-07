---
title: Classification Models and Their Performance
notebook: Classification Models and Their Performance.ipynb
nav_include: 5
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('white')
sns.set_context('paper', font_scale=1.5)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')
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




```python
def score(model, X_train, y_train, X_test, y_test):
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    test_class0 = model.score(X_test[y_test==0], y_test[y_test==0])
    test_class1 = model.score(X_test[y_test==1], y_test[y_test==1])
    test_class2 = model.score(X_test[y_test==2], y_test[y_test==2])
    return pd.Series([train_acc, test_acc, test_class0, test_class1, test_class2],
                    index = ['Train accuracy', 'Test accuracy', 
                             "Test accuracy CN", "Test accuracy CI", "Test accuracy AD"])
```


## Baseline Model

We used `stratified` strategy to generate predictions as a simple baseline to compare with other classifiers we learned in class.



```python
dc = DummyClassifier(strategy='stratified', random_state=9001)
dc.fit(X_train,y_train)
print('Dummy Classifier Training Score: ', dc.score(X_train,y_train))
print('Dummy Classifier Test Score: ', dc.score(X_test,y_test))
print('Dummy Classifier Confusion Matrix:\n', confusion_matrix(y_test,dc.predict(X_test)))
dc_score = score(dc, X_train, y_train, X_test, y_test)
```


    Dummy Classifier Training Score:  0.423510466989
    Dummy Classifier Test Score:  0.444444444444
    Dummy Classifier Confusion Matrix:
     [[14 19  9]
     [27 52 14]
     [ 7 14  6]]


## Logistic Regression

We tested 6 kinds of logistic regression, logistic regression with l1 penalty, logistic regression with l2 penalty, unweighted logistic regression, weighted logistic regression, one-vs-rest logistic regression and multinomial logistic regression. We chose the best parameters with cross validation. We found that a large regularization term is needed for all classifiers, indicating that we have too many variables.



```python
#l1
log_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear', random_state=9001)
log_l1.fit(X_train,y_train)

#l2
log_l2 = LogisticRegressionCV(penalty = 'l2', random_state=9001)
log_l2.fit(X_train,y_train)

#Unweighted logistic regression
unweighted_logistic = LogisticRegressionCV(random_state=9001)
unweighted_logistic.fit(X_train,y_train)

#Weighted logistic regression
weighted_logistic = LogisticRegressionCV(class_weight='balanced', random_state=9001)
weighted_logistic.fit(X_train,y_train)

#ovr
log_ovr = LogisticRegressionCV(multi_class = 'ovr', random_state=9001)
log_ovr.fit(X_train,y_train)

#multinomial
log_multinomial = LogisticRegressionCV(multi_class = 'multinomial', solver = 'newton-cg', random_state=9001)
log_multinomial.fit(X_train,y_train)

print("Regularization strength: ")
print("-------------------------")
print("Logistic regression with l1 penalty:", log_l1.C_[0])
print("Logistic regression with l2 penalty:", log_l2.C_[0])
print("Unweighted logistic regression: ", unweighted_logistic.C_[0])
print("Weighted logistic regression: ", weighted_logistic.C_[0])
print("OVR logistic regression: ", log_ovr.C_[0])
print("Multinomial logistic regression: ", log_multinomial.C_[0])
```


    Regularization strength: 
    -------------------------
    Logistic regression with l1 penalty: 0.35938136638
    Logistic regression with l2 penalty: 0.35938136638
    Unweighted logistic regression:  0.35938136638
    Weighted logistic regression:  0.00599484250319
    OVR logistic regression:  0.35938136638
    Multinomial logistic regression:  2.78255940221




```python
#Computing the score on the train set - 
print("Training accuracy")
print("-------------------------------------------------")
print('Logistic Regression with l1 penalty train Score: ',log_l1.score(X_train, y_train))
print('Logistic Regression with l2 penalty train Score: ',log_l2.score(X_train, y_train))
print('Unweighted Logistic Regression with train Score: ',unweighted_logistic.score(X_train, y_train))
print('Weighted Logistic Regression train Score: ',weighted_logistic.score(X_train, y_train))
print('OVR Logistic Regression train Score: ',log_ovr.score(X_train, y_train))
print('Multinomial Logistic Regression train Score: ',log_multinomial.score(X_train, y_train))

print('\n')

#Computing the score on the test set - 
print("Test accuracy")
print("-------------------------------------------------")
print('Logistic Regression with l1 penalty test Score: ',log_l1.score(X_test, y_test))
print('Logistic Regression with l2 penalty test Score: ',log_l2.score(X_test, y_test))
print('Unweighted Logistic Regression test Score: ',unweighted_logistic.score(X_test, y_test))
print('Weighted Logistic Regression test Score: ',weighted_logistic.score(X_test, y_test))
print('OVR Logistic Regression test Score: ',log_ovr.score(X_test, y_test))
print('Multinomial Logistic Regression test Score: ',log_multinomial.score(X_test, y_test))
```


    Training accuracy
    -------------------------------------------------
    Logistic Regression with l1 penalty train Score:  0.830917874396
    Logistic Regression with l2 penalty train Score:  0.610305958132
    Unweighted Logistic Regression with train Score:  0.610305958132
    Weighted Logistic Regression train Score:  0.471819645733
    OVR Logistic Regression train Score:  0.610305958132
    Multinomial Logistic Regression train Score:  0.845410628019
    
    
    Test accuracy
    -------------------------------------------------
    Logistic Regression with l1 penalty test Score:  0.827160493827
    Logistic Regression with l2 penalty test Score:  0.592592592593
    Unweighted Logistic Regression test Score:  0.592592592593
    Weighted Logistic Regression test Score:  0.407407407407
    OVR Logistic Regression test Score:  0.592592592593
    Multinomial Logistic Regression test Score:  0.796296296296




```python
l1_score = score(log_l1, X_train, y_train, X_test, y_test)
l2_score = score(log_l2, X_train, y_train, X_test, y_test)
weighted_score = score(weighted_logistic, X_train, y_train, X_test, y_test)
unweighted_score = score(unweighted_logistic, X_train, y_train, X_test, y_test)
ovr_score = score(log_ovr, X_train, y_train, X_test, y_test)
multi_score = score(log_multinomial, X_train, y_train, X_test, y_test)
```




```python
l1_pred = log_l1.predict(X_test)
l2_pred = log_l2.predict(X_test)
weighted_pred = weighted_logistic.predict(X_test)
unweighted_pred = unweighted_logistic.predict(X_test)
ovr_pred = log_ovr.predict(X_test)
multi_pred = log_multinomial.predict(X_test)

print("Confusion Matrix")
print("Logistic Regression with l1 penalty:\n",
      confusion_matrix(y_test, l1_pred))
print("Logistic Regression with l2 penalty:\n",
      confusion_matrix(y_test, l2_pred))
print("Unweighted Logistic Regression:\n",
      confusion_matrix(y_test, unweighted_pred))
print("Weighted Logistic Regression:\n",
      confusion_matrix(y_test, weighted_pred))
print("OVR Logistic Regression:\n",
      confusion_matrix(y_test, ovr_pred))
print("Multinomial Logistic Regression:\n",
      confusion_matrix(y_test, multi_pred))
```


    Confusion Matrix
    Logistic Regression with l1 penalty:
     [[27 15  0]
     [ 5 84  4]
     [ 0  4 23]]
    Logistic Regression with l2 penalty:
     [[ 0 42  0]
     [ 0 86  7]
     [ 0 17 10]]
    Unweighted Logistic Regression:
     [[ 0 42  0]
     [ 0 86  7]
     [ 0 17 10]]
    Weighted Logistic Regression:
     [[33  6  3]
     [53 14 26]
     [ 2  6 19]]
    OVR Logistic Regression:
     [[ 0 42  0]
     [ 0 86  7]
     [ 0 17 10]]
    Multinomial Logistic Regression:
     [[28 14  0]
     [10 78  5]
     [ 0  4 23]]


## Discriminant Analysis

We performed normalization on continuous predictors and used Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) as our models. LDA performs really well.



```python
cols_standardize = [
    c for c in X_train.columns 
    if (not c.startswith('PT')) \
        or (c=='PTEDUCAT') or (c=='PTAGE')]

X_train_std = X_train.copy()
X_test_std = X_test.copy()
for c in cols_standardize:
    col_mean = np.mean(X_train[c])
    col_sd = np.std(X_train[c])
    if col_sd > (1e-10)*col_mean:
        X_train_std[c] = (X_train[c]-col_mean)/col_sd
        X_test_std[c] = (X_test[c]-col_mean)/col_sd
```




```python
X_train_std.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PTAGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>PTRACCAT_Asian</th>
      <th>PTRACCAT_Black</th>
      <th>PTRACCAT_Hawaiian/Other_PI</th>
      <th>PTRACCAT_More_than_one</th>
      <th>PTRACCAT_Unknown</th>
      <th>PTRACCAT_White</th>
      <th>PTETHCAT_Not_Hisp/Latino</th>
      <th>PTMARRY_Married</th>
      <th>PTMARRY_Never_married</th>
      <th>PTMARRY_Widowed</th>
      <th>APOE4</th>
      <th>CSF_ABETA</th>
      <th>CSF_TAU</th>
      <th>CSF_PTAU</th>
      <th>FDG</th>
      <th>FDG_slope</th>
      <th>AV45</th>
      <th>AV45_slope</th>
      <th>ADAS13</th>
      <th>ADAS13_slope</th>
      <th>MMSE</th>
      <th>MMSE_slope</th>
      <th>RAVLT_immediate</th>
      <th>RAVLT_immediate_slope</th>
      <th>RAVLT_learning</th>
      <th>RAVLT_learning_slope</th>
      <th>RAVLT_forgetting</th>
      <th>RAVLT_forgetting_slope</th>
      <th>RAVLT_perc_forgetting</th>
      <th>RAVLT_perc_forgetting_slope</th>
      <th>MOCA</th>
      <th>MOCA_slope</th>
      <th>EcogPtMem</th>
      <th>EcogPtMem_slope</th>
      <th>EcogPtLang</th>
      <th>EcogPtLang_slope</th>
      <th>EcogPtVisspat</th>
      <th>EcogPtVisspat_slope</th>
      <th>EcogPtPlan</th>
      <th>EcogPtPlan_slope</th>
      <th>EcogPtOrgan</th>
      <th>EcogPtOrgan_slope</th>
      <th>EcogPtDivatt</th>
      <th>EcogPtDivatt_slope</th>
      <th>EcogSPMem</th>
      <th>EcogSPMem_slope</th>
      <th>EcogSPLang</th>
      <th>EcogSPLang_slope</th>
      <th>EcogSPVisspat</th>
      <th>EcogSPVisspat_slope</th>
      <th>EcogSPPlan</th>
      <th>EcogSPPlan_slope</th>
      <th>EcogSPOrgan</th>
      <th>EcogSPOrgan_slope</th>
      <th>EcogSPDivatt</th>
      <th>EcogSPDivatt_slope</th>
      <th>FAQ</th>
      <th>FAQ_slope</th>
      <th>Ventricles</th>
      <th>Ventricles_slope</th>
      <th>Hippocampus</th>
      <th>Hippocampus_slope</th>
      <th>WholeBrain</th>
      <th>WholeBrain_slope</th>
      <th>Entorhinal</th>
      <th>Entorhinal_slope</th>
      <th>Fusiform</th>
      <th>Fusiform_slope</th>
      <th>MidTemp</th>
      <th>MidTemp_slope</th>
      <th>ICV</th>
      <th>ICV_slope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.208480</td>
      <td>0</td>
      <td>-2.852257</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.823084</td>
      <td>-1.421715</td>
      <td>1.137347</td>
      <td>-0.321586</td>
      <td>-0.833130</td>
      <td>0.811782</td>
      <td>1.257634</td>
      <td>0.023763</td>
      <td>2.140034</td>
      <td>0.236256</td>
      <td>-2.742373</td>
      <td>-0.471665</td>
      <td>-0.010026</td>
      <td>-0.001926</td>
      <td>0.003844</td>
      <td>-0.000699</td>
      <td>0.028414</td>
      <td>-0.018332</td>
      <td>0.019484</td>
      <td>-0.012471</td>
      <td>-0.003150</td>
      <td>-0.004052</td>
      <td>-0.827740</td>
      <td>1.136728</td>
      <td>-0.981277</td>
      <td>-0.002156</td>
      <td>-0.715234</td>
      <td>-0.030329</td>
      <td>-0.713577</td>
      <td>-0.002410</td>
      <td>-0.843488</td>
      <td>-0.007633</td>
      <td>-1.076964</td>
      <td>-0.016544</td>
      <td>1.929076</td>
      <td>-0.088208</td>
      <td>-0.844339</td>
      <td>0.954666</td>
      <td>1.066767</td>
      <td>1.048123</td>
      <td>2.524016</td>
      <td>0.060641</td>
      <td>2.287022</td>
      <td>0.021430</td>
      <td>2.330806</td>
      <td>-0.631242</td>
      <td>2.934694</td>
      <td>0.124173</td>
      <td>-0.205919</td>
      <td>0.381384</td>
      <td>-1.351616</td>
      <td>0.022285</td>
      <td>-1.761500</td>
      <td>-0.567555</td>
      <td>-0.820814</td>
      <td>-1.269796</td>
      <td>-1.426968</td>
      <td>0.156847</td>
      <td>-2.102069</td>
      <td>-0.192827</td>
      <td>-1.574482</td>
      <td>0.093937</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.759714</td>
      <td>1</td>
      <td>1.376909</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.703236</td>
      <td>-0.567568</td>
      <td>-0.086517</td>
      <td>-0.128000</td>
      <td>0.602793</td>
      <td>2.864460</td>
      <td>1.903103</td>
      <td>1.199280</td>
      <td>-0.210782</td>
      <td>-0.364551</td>
      <td>0.586715</td>
      <td>0.595629</td>
      <td>0.202187</td>
      <td>-0.189405</td>
      <td>0.884055</td>
      <td>-0.907178</td>
      <td>-0.118596</td>
      <td>1.750365</td>
      <td>-0.556812</td>
      <td>1.770545</td>
      <td>0.227228</td>
      <td>-0.044392</td>
      <td>-0.827740</td>
      <td>-0.229483</td>
      <td>-1.153377</td>
      <td>-0.000458</td>
      <td>-0.444998</td>
      <td>-0.101471</td>
      <td>-0.713577</td>
      <td>-0.065235</td>
      <td>-0.843488</td>
      <td>-0.064451</td>
      <td>-1.076964</td>
      <td>-0.072474</td>
      <td>0.573694</td>
      <td>-0.738447</td>
      <td>-0.106940</td>
      <td>-1.151805</td>
      <td>0.487637</td>
      <td>-1.687565</td>
      <td>1.262926</td>
      <td>-2.477232</td>
      <td>0.148179</td>
      <td>-1.526413</td>
      <td>0.971872</td>
      <td>-1.384981</td>
      <td>-0.484392</td>
      <td>-0.424658</td>
      <td>-0.053348</td>
      <td>-0.251084</td>
      <td>0.483644</td>
      <td>-0.306894</td>
      <td>-0.134464</td>
      <td>-0.028641</td>
      <td>-0.070387</td>
      <td>0.188014</td>
      <td>0.721399</td>
      <td>-0.067438</td>
      <td>0.019784</td>
      <td>0.506511</td>
      <td>-0.489132</td>
      <td>-0.265646</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.257208</td>
      <td>0</td>
      <td>0.607970</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.823084</td>
      <td>-0.775573</td>
      <td>-1.038657</td>
      <td>-0.920353</td>
      <td>0.572072</td>
      <td>0.019993</td>
      <td>0.547072</td>
      <td>0.023763</td>
      <td>-0.798486</td>
      <td>-0.710060</td>
      <td>0.586715</td>
      <td>0.463914</td>
      <td>1.183334</td>
      <td>1.011032</td>
      <td>0.884055</td>
      <td>-0.590705</td>
      <td>0.280068</td>
      <td>-1.497230</td>
      <td>-0.495770</td>
      <td>-1.722958</td>
      <td>0.452935</td>
      <td>0.701979</td>
      <td>-0.530055</td>
      <td>-0.049247</td>
      <td>-0.464974</td>
      <td>-0.085273</td>
      <td>-0.715234</td>
      <td>-0.033968</td>
      <td>-0.713577</td>
      <td>-0.065235</td>
      <td>-0.018265</td>
      <td>0.055382</td>
      <td>-0.053520</td>
      <td>0.069781</td>
      <td>-1.052764</td>
      <td>0.042214</td>
      <td>-0.844339</td>
      <td>-0.077852</td>
      <td>-0.670610</td>
      <td>-0.211993</td>
      <td>-0.754818</td>
      <td>0.111691</td>
      <td>-0.824015</td>
      <td>-0.212459</td>
      <td>-0.930635</td>
      <td>0.246899</td>
      <td>-0.647205</td>
      <td>-0.415515</td>
      <td>-0.294977</td>
      <td>-0.883418</td>
      <td>-1.104796</td>
      <td>0.020857</td>
      <td>-1.300396</td>
      <td>0.310720</td>
      <td>0.456478</td>
      <td>-0.560840</td>
      <td>0.292776</td>
      <td>0.016824</td>
      <td>-0.650452</td>
      <td>0.224140</td>
      <td>-1.239633</td>
      <td>-0.014198</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.521887</td>
      <td>0</td>
      <td>-0.160970</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.823084</td>
      <td>-0.445863</td>
      <td>0.222762</td>
      <td>0.322200</td>
      <td>0.441716</td>
      <td>0.850276</td>
      <td>-0.548541</td>
      <td>-0.049974</td>
      <td>-0.994388</td>
      <td>-0.282924</td>
      <td>0.586715</td>
      <td>0.519651</td>
      <td>0.730497</td>
      <td>0.190865</td>
      <td>-0.546632</td>
      <td>0.454839</td>
      <td>-0.915923</td>
      <td>1.164759</td>
      <td>-1.032937</td>
      <td>0.965802</td>
      <td>0.678642</td>
      <td>0.204009</td>
      <td>0.387783</td>
      <td>0.167724</td>
      <td>0.223444</td>
      <td>0.375527</td>
      <td>-0.174781</td>
      <td>0.049689</td>
      <td>-0.007973</td>
      <td>0.191936</td>
      <td>0.531879</td>
      <td>0.404523</td>
      <td>1.652218</td>
      <td>0.250482</td>
      <td>-0.103997</td>
      <td>-0.010217</td>
      <td>-0.844339</td>
      <td>-0.186429</td>
      <td>-0.670610</td>
      <td>-0.271617</td>
      <td>-0.754818</td>
      <td>-0.343706</td>
      <td>-0.824015</td>
      <td>-0.076909</td>
      <td>-0.115275</td>
      <td>0.065754</td>
      <td>-0.484392</td>
      <td>-0.473929</td>
      <td>-0.005181</td>
      <td>-0.032439</td>
      <td>0.013960</td>
      <td>0.020857</td>
      <td>-0.000094</td>
      <td>-0.003749</td>
      <td>0.006635</td>
      <td>-0.003683</td>
      <td>0.010325</td>
      <td>0.015345</td>
      <td>0.018697</td>
      <td>0.004091</td>
      <td>-0.005136</td>
      <td>0.004314</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.121905</td>
      <td>1</td>
      <td>-0.160970</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.823084</td>
      <td>-1.521292</td>
      <td>0.516578</td>
      <td>0.056582</td>
      <td>0.613315</td>
      <td>1.113319</td>
      <td>1.781157</td>
      <td>0.338415</td>
      <td>-0.602585</td>
      <td>-0.189067</td>
      <td>0.586715</td>
      <td>0.476561</td>
      <td>-0.099704</td>
      <td>0.488832</td>
      <td>-0.546632</td>
      <td>0.189374</td>
      <td>-0.118596</td>
      <td>0.780831</td>
      <td>-0.160039</td>
      <td>0.662249</td>
      <td>-0.449892</td>
      <td>0.202553</td>
      <td>-0.133156</td>
      <td>0.119957</td>
      <td>-0.809176</td>
      <td>0.038450</td>
      <td>-0.444998</td>
      <td>-0.146690</td>
      <td>-0.360775</td>
      <td>-0.127391</td>
      <td>0.806959</td>
      <td>-0.571339</td>
      <td>0.287627</td>
      <td>-0.263162</td>
      <td>0.709232</td>
      <td>0.108071</td>
      <td>-0.401908</td>
      <td>0.043380</td>
      <td>-0.273501</td>
      <td>0.072295</td>
      <td>0.506272</td>
      <td>-0.367695</td>
      <td>1.120385</td>
      <td>-0.514067</td>
      <td>-0.115275</td>
      <td>0.347360</td>
      <td>-0.484392</td>
      <td>-0.455347</td>
      <td>-0.005181</td>
      <td>-0.032439</td>
      <td>1.295160</td>
      <td>0.219009</td>
      <td>-0.000094</td>
      <td>-0.003749</td>
      <td>0.006635</td>
      <td>-0.003683</td>
      <td>0.010325</td>
      <td>0.015345</td>
      <td>0.018697</td>
      <td>0.004091</td>
      <td>1.652198</td>
      <td>-0.047345</td>
    </tr>
  </tbody>
</table>
</div>





```python
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train_std,y_train)
qda.fit(X_train_std,y_train)

print("Training accuracy")
print("------------------")
print('LDA Train Score: ',lda.score(X_train_std,y_train))
print('QDA Train Score: ',qda.score(X_train_std,y_train))

print('\n')

print("Test accuracy")
print("------------------")
print('LDA Test Score: ',lda.score(X_test_std,y_test))
print('QDA Test Score: ',qda.score(X_test_std,y_test))
```


    Training accuracy
    ------------------
    LDA Train Score:  0.848631239936
    QDA Train Score:  0.819645732689
    
    
    Test accuracy
    ------------------
    LDA Test Score:  0.814814814815
    QDA Test Score:  0.697530864198




```python
lda_score = score(lda, X_train_std, y_train, X_test_std, y_test)
qda_score = score(qda, X_train_std, y_train, X_test_std, y_test)
```




```python
lda_pred = lda.predict(X_test_std)
qda_pred = qda.predict(X_test_std)

print("Confusion Matrix")
print("LDA:\n",
      confusion_matrix(y_test, lda_pred))
print("QDA:\n",
      confusion_matrix(y_test, qda_pred))
```


    Confusion Matrix
    LDA:
     [[30 12  0]
     [10 78  5]
     [ 1  2 24]]
    QDA:
     [[23 18  1]
     [ 6 70 17]
     [ 0  7 20]]


## K-Nearest Neighbours

The optimal number of neighbours is 41, which is a relatively large number considering that we only have 783 observations. The accuracy is not satisfactory as well.



```python
cv_fold = KFold(n_splits=5, shuffle=True, random_state=9001)

max_score = 0
max_k = 0 

for k in range(1,60):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn_val_score = cross_val_score(knn, X_train, y_train, cv=cv_fold).mean()
    if knn_val_score > max_score:
        max_k = k
        max_score = knn_val_score
        
knn = KNeighborsClassifier(n_neighbors = max_k)
knn.fit(X_train,y_train)

print("Optimal number of neighbours: ", max_k)
print('KNN Training Accuracy: ', knn.score(X_train,y_train))
print('KNN Test Accuracy: ', knn.score(X_test,y_test))

knn_score = score(knn, X_train, y_train, X_test, y_test)
```


    Optimal number of neighbours:  41
    KNN Training Accuracy:  0.566827697262
    KNN Test Accuracy:  0.574074074074




```python
knn_pred = knn.predict(X_test)

print("KNN Confusion Matrix:\n",
      confusion_matrix(y_test, knn_pred))
```


    KNN Confusion Matrix:
     [[ 0 42  0]
     [ 0 92  1]
     [ 1 25  1]]


## Decision Tree

We used 5-fold cross validation to find the optimal depth for the decision tree. The optimal depth is 6.



```python
depth = []
for i in range(3,20):
    dt = DecisionTreeClassifier(max_depth=i, random_state=9001)
    # Perform 5-fold cross validation 
    scores = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=cv_fold, n_jobs=-1)
    depth.append((i, scores.mean(), scores.std())) 
depthvals = [t[0] for t in depth]
cvmeans = np.array([t[1] for t in depth])
cvstds = np.array([t[2] for t in depth])
max_indx = np.argmax(cvmeans)
md_best = depthvals[max_indx]
print('Optimal depth:',md_best)
dt_best = DecisionTreeClassifier(max_depth=md_best, random_state=9001)
dt_best.fit(X_train, y_train).score(X_test, y_test)
dt_score = score(dt_best, X_train, y_train, X_test, y_test)
```


    Optimal depth: 6




```python
print('Decision Tree Training Accuracy: ', dt_best.score(X_train,y_train))
print('Decision Tree Test Accuracy: ', dt_best.score(X_test,y_test))
```


    Decision Tree Training Accuracy:  0.90499194847
    Decision Tree Test Accuracy:  0.746913580247




```python
dt_pred = dt_best.predict(X_test)

print("Decision Tree Confusion Matrix:\n",
      confusion_matrix(y_test, dt_pred))
```


    Decision Tree Confusion Matrix:
     [[24 18  0]
     [19 73  1]
     [ 0  3 24]]


## Random Forest

We used `GridSearchCV` to find the optimal number of trees and tree depth. We then used the optimal value to perform random forest classification.



```python
trees = [2**x for x in range(8)]  # 1, 2, 4, 8, 16, 32, ...
depth = [2, 4, 6, 8, 10, 12, 14, 16]
parameters = {'n_estimators': trees,
              'max_depth': depth}
rf = RandomForestClassifier(random_state=9001)
rf_cv = GridSearchCV(rf, parameters, cv=cv_fold)
rf_cv.fit(X_train, y_train)
best_score = np.argmax(rf_cv.cv_results_['mean_test_score'])
result = rf_cv.cv_results_['params'][best_score]
opt_depth = result['max_depth']
opt_tree = result['n_estimators']
print("Optimal number of trees {}, tree depth: {}".format(opt_tree, opt_depth))
rf = RandomForestClassifier(n_estimators=opt_tree, max_depth=opt_depth, random_state=9001)
rf.fit(X_train, y_train)
print('\n')
print('Random Forest Training Accuracy: ', rf.score(X_train,y_train))
print('Random Forest Test Accuracy: ', rf.score(X_test,y_test))
rf_score = score(rf, X_train, y_train, X_test, y_test)
```


    Optimal number of trees 32, tree depth: 12
    
    
    Random Forest Training Accuracy:  0.998389694042
    Random Forest Test Accuracy:  0.802469135802




```python
rf_pred = rf.predict(X_test)

print("Random Forest Confusion Matrix:\n",
      confusion_matrix(y_test, rf_pred))
```


    Random Forest Confusion Matrix:
     [[20 22  0]
     [ 5 85  3]
     [ 0  2 25]]


## AdaBoost

We used the optimal tree depth found by cross validation in the decision tree classifier, and performed `GridSearchCV` to find the optimal number of trees and learning rate.



```python
trees = [2**x for x in range(6)]  # 1, 2, 4, 8, 16, 32, ...
learning_rate = [0.1, 0.5, 1, 5]
parameters = {'n_estimators': trees,
              'learning_rate': learning_rate}
ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=md_best),
                        random_state=9001)
ab_cv = GridSearchCV(ab, parameters, cv=cv_fold)
ab_cv.fit(X_train, y_train)
best_score = np.argmax(ab_cv.cv_results_['mean_test_score'])
result = ab_cv.cv_results_['params'][best_score]
opt_learning_rate = result['learning_rate']
opt_tree = result['n_estimators']
print("Optimal number of trees {}, learning rate: {}".format(opt_tree, opt_learning_rate))
ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=md_best), n_estimators=opt_tree,
                       learning_rate=opt_learning_rate, random_state=9001)
ab.fit(X_train, y_train)
print('\n')
print('AdaBoost Training Accuracy: ', ab.score(X_train,y_train))
print('AdaBoost Test Accuracy: ', ab.score(X_test,y_test))
ab_score = score(ab, X_train, y_train, X_test, y_test)
```


    Optimal number of trees 16, learning rate: 1
    
    
    AdaBoost Training Accuracy:  1.0
    AdaBoost Test Accuracy:  0.740740740741




```python
ab_pred = ab.predict(X_test)

print("AdaBoost Confusion Matrix:\n",
      confusion_matrix(y_test, ab_pred))
```


    AdaBoost Confusion Matrix:
     [[20 22  0]
     [13 76  4]
     [ 0  3 24]]



## Performance Summary



```python
score_df = pd.DataFrame({'Dummy Classifier': dc_score,
                         'Logistic Regression with l1': l1_score, 
                         'Logistic Regression with l2': l2_score,
                         'Weighted logistic': weighted_score,
                         'Unweighted logistic': unweighted_score,
                         'OVR': ovr_score,
                         'Multinomial': multi_score,
                         'KNN': knn_score,
                         'LDA': lda_score,
                         'QDA': qda_score,
                         'Decision Tree': dt_score,
                         'Random Forest': rf_score,
                         'AdaBoost': ab_score})
score_df
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AdaBoost</th>
      <th>Decision Tree</th>
      <th>Dummy Classifier</th>
      <th>KNN</th>
      <th>LDA</th>
      <th>Logistic Regression with l1</th>
      <th>Logistic Regression with l2</th>
      <th>Multinomial</th>
      <th>OVR</th>
      <th>QDA</th>
      <th>Random Forest</th>
      <th>Unweighted logistic</th>
      <th>Weighted logistic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train accuracy</th>
      <td>1.000000</td>
      <td>0.904992</td>
      <td>0.423510</td>
      <td>0.566828</td>
      <td>0.848631</td>
      <td>0.830918</td>
      <td>0.610306</td>
      <td>0.845411</td>
      <td>0.610306</td>
      <td>0.819646</td>
      <td>0.998390</td>
      <td>0.610306</td>
      <td>0.471820</td>
    </tr>
    <tr>
      <th>Test accuracy</th>
      <td>0.740741</td>
      <td>0.746914</td>
      <td>0.444444</td>
      <td>0.574074</td>
      <td>0.814815</td>
      <td>0.827160</td>
      <td>0.592593</td>
      <td>0.796296</td>
      <td>0.592593</td>
      <td>0.697531</td>
      <td>0.802469</td>
      <td>0.592593</td>
      <td>0.407407</td>
    </tr>
    <tr>
      <th>Test accuracy CN</th>
      <td>0.476190</td>
      <td>0.571429</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.714286</td>
      <td>0.642857</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.547619</td>
      <td>0.476190</td>
      <td>0.000000</td>
      <td>0.785714</td>
    </tr>
    <tr>
      <th>Test accuracy CI</th>
      <td>0.817204</td>
      <td>0.784946</td>
      <td>0.516129</td>
      <td>0.989247</td>
      <td>0.838710</td>
      <td>0.903226</td>
      <td>0.924731</td>
      <td>0.838710</td>
      <td>0.924731</td>
      <td>0.752688</td>
      <td>0.913978</td>
      <td>0.924731</td>
      <td>0.150538</td>
    </tr>
    <tr>
      <th>Test accuracy AD</th>
      <td>0.888889</td>
      <td>0.888889</td>
      <td>0.222222</td>
      <td>0.037037</td>
      <td>0.888889</td>
      <td>0.851852</td>
      <td>0.370370</td>
      <td>0.851852</td>
      <td>0.370370</td>
      <td>0.740741</td>
      <td>0.925926</td>
      <td>0.370370</td>
      <td>0.703704</td>
    </tr>
  </tbody>
</table>
</div>





```python
names_sorted = [
    pair[0] for pair in sorted(
        zip(score_df.columns, score_df.loc['Test accuracy']), 
        key=lambda x:x[1], reverse=False) ]
names_ticks = [
    n.replace(' ','\n') for n in names_sorted ]
```




```python
bar_width = 0.3
plt.figure(figsize = (16,10))
for i,idx in enumerate(score_df.index[:2]):
    plt.bar(np.arange(len(names_sorted))+i*bar_width, score_df[names_sorted].loc[idx], 
            bar_width, color=sns.color_palette()[i], label=idx)
plt.xticks(np.arange(len(names_sorted))+0.15, names_ticks)
plt.ylabel('Overall Classification Accuracy')
plt.xlabel('Classifiers')
plt.legend(loc='best')
plt.ylim(0,1)
plt.show()
```



![png](Classification%20Models%20and%20Their%20Performance_files/Classification%20Models%20and%20Their%20Performance_41_0.png)




```python
bar_width = 0.3
plt.figure(figsize = (16,10))
for i,idx in enumerate(score_df.index[2:]):
    plt.bar(1.2*np.arange(len(names_sorted))+i*bar_width, score_df[names_sorted].loc[idx], 
            bar_width, color=sns.color_palette()[i+2], label=idx)
plt.xticks(1.2*np.arange(len(names_sorted))+0.3, names_ticks)
plt.ylabel('Classification Accuracy on Each Class')
plt.xlabel('Classifiers')
plt.legend(loc='best')
plt.ylim(0,1)
plt.show()
```



![png](Classification%20Models%20and%20Their%20Performance_files/Classification%20Models%20and%20Their%20Performance_42_0.png)


For baseline models, people usually use the dummy classifier with "stratified" strategy or the "most frequent" strategy. The "stratified" method generates prediction according to the class distribution of the training set. The "most frequent" strategy always predicts the most frequent class. We adopted the `stratified` strategy implemented by `sklearn`'s `DummyClassifer` as the baseline model. As can be seen above, all other classification models we used outperformed the dummy classifier as expected.

Based on the above summary, `AdaBoost` and `Random Forest` have perfect training accuracy, 100%. Three classifiers with the highest test accuracies are `Random Forest` (0.802469), `LDA` (0.814815) and `Logistic Regression with l1` (0.827160), among which `Logistic Regression with l1` has the best performance.

For classifying `CN` patients, weighted logistic regression has the highest test accuracy (0.833333), so it performs the best for determining Cognitively Normal patients. However, KNN, logistic regression with l2 regularization, OvR logistic regression and unweighted logistic regression have zero accuracy on classifying `CN` patients. Since all of them have very high accuracy on `CI` but low accuracy on `AD`, we think these four models probably classify almost all the `CN` patients into `CI` (as can been seen in the confusion matrix), leading to zero accuracy on `CN` and high accuracy on `CI`.

KNN has the highest test accuracy (0.989247) on diagnosing `CI` cognitive impairment patients. Logistic regression with l2 regularization, random forest classifier, OvR logistic regression and unweighted logistic regression all reach 0.9 accuracy on diagnosing `CI` patients.

Since we focus on the diagnosis of Alzheimer's disease, we are more concerned about the test accuracy on `AD` patients. Random forest classifier has the highest test accuracy (0.925926) on `AD` patients. Adaboost, LDA, decision tree, and multinomial logistic regression all achieve test accuracy reaching 0.90 on the classification of `AD`.

In addition, we find an interesting pattern in the above barplots of accuracy. There seems to be three types of classifiers. Type I includes `Weighted Logistic` and `Dummy Classifier`. Their overall accuracies are at a relatively low level around 0.40. Type II includes `KNN`, `Logistic Regression with l2`, `OvR` and `Unweighted Logistic`. Their overall accuracies are at a midium level around 0.60, and their partial accuracies on the three classes are very unbalanced. None of them can predict correctly on `CN`. Type III includes `QDA`, `AdaBoost`, `Decision Tree`, `Multinomial`, `Random Forest`, `LDA` and `Logistic Regression with l1`. Their overall accuracies are at a relatively high level over 0.70, and their partial accuracies on the three classes are basically balanced. Every Type III classifier has its own advantage and can be competitive substitution to each other.

To conclude, `Logistic Regression with l1`, `LDA` and `Random Forest` perform the best if we are concerned about both overall test accuracy and correctly diagnosing `AD` patients. Other models such as `QDA`, `AdaBoost`, `Decision Tree` and `Multinomial` are also promising.
