---
title: Higher-level Models
notebook: Higher-level Models.ipynb
nav_include: 7
---

## Contents
{:.no_toc}
*  
{: toc}

## Neural Networks

To obtain higher classification accuracy, we implemented neural networks, which we did not learn in the class. Multi-layer perceptrons neural network is a supervised method, and is very powerful in classifying Alzheimer's disease as shown below.



```python
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')
```




```python
df_train = pd.read_csv("data/ADNIMERGE_train.csv")
df_test = pd.read_csv("data/ADNIMERGE_test.csv")
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
print(X_train_std.shape)
X_train_std.head()
```


    (621, 75)





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
cv_fold = KFold(n_splits=3, shuffle=True, random_state=9001)
parameters = {'alpha': [1e-3, 1e-2, 1e-1, 0.3, 1, 3, 1e1, 1e2, 1e3],
              'hidden_layer_sizes': [(50), (100), (200), (500), 
                                     (50, 10), (50, 25), 
                                     (100, 10)]}
mlp = MLPClassifier(solver='lbfgs', activation='logistic', random_state=9001)
mlp_cv = GridSearchCV(mlp, parameters, cv=cv_fold)
mlp_cv.fit(X_train_std, y_train)
best_score = np.argmax(mlp_cv.cv_results_['mean_test_score'])
result = mlp_cv.cv_results_['params'][best_score]
a = result['alpha']
hidden_layer = result['hidden_layer_sizes']
mlp = MLPClassifier(solver='lbfgs', activation='logistic', random_state=9001,
                    alpha = a, hidden_layer_sizes=hidden_layer)
mlp = mlp.fit(X_train_std, y_train)
```




```python
print("Optimal parameters")
print("L2 penalty parameter: ", a)
print("Hidden Layer Sizes: ", hidden_layer)
print('\n-----------------\n')
print("Training accuracy: ", mlp.score(X_train_std, y_train))
print("Test accuracy: ", mlp.score(X_test_std, y_test))
print('\n-----------------\n')
print('Test Confusion Matrix: ')
print(confusion_matrix(y_test, mlp.predict(X_test_std)))
nn_score = score(mlp, X_train_std, y_train, X_test_std, y_test)
```


    Optimal parameters
    L2 penalty parameter:  3
    Hidden Layer Sizes:  200
    
    -----------------
    
    Training accuracy:  0.890499194847
    Test accuracy:  0.827160493827
    
    -----------------
    
    Test Confusion Matrix: 
    [[29 13  0]
     [ 7 83  3]
     [ 0  5 22]]




```python
rf_best = RandomForestClassifier(n_estimators=32, max_depth=12, random_state=9001)
rf_best.fit(X_train, y_train)
rf_score = score(rf_best, X_train, y_train, X_test, y_test)
print('\n-----------------\n')
print("Training accuracy: ", rf_best.score(X_train, y_train))
print("Test accuracy: ", rf_best.score(X_test, y_test))
print('\n-----------------\n')
print('Test Confusion Matrix: ')
print(confusion_matrix(y_test, rf_best.predict(X_test)))
```


    
    -----------------
    
    Training accuracy:  0.998389694042
    Test accuracy:  0.802469135802
    
    -----------------
    
    Test Confusion Matrix: 
    [[20 22  0]
     [ 5 85  3]
     [ 0  2 25]]




```python
score_df = pd.DataFrame({"Neural Network": nn_score,
                         "Random Forest": rf_score})
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
      <th>Neural Network</th>
      <th>Random Forest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train accuracy</th>
      <td>0.890499</td>
      <td>0.998390</td>
    </tr>
    <tr>
      <th>Test accuracy</th>
      <td>0.827160</td>
      <td>0.802469</td>
    </tr>
    <tr>
      <th>Test accuracy CN</th>
      <td>0.690476</td>
      <td>0.476190</td>
    </tr>
    <tr>
      <th>Test accuracy CI</th>
      <td>0.892473</td>
      <td>0.913978</td>
    </tr>
    <tr>
      <th>Test accuracy AD</th>
      <td>0.814815</td>
      <td>0.925926</td>
    </tr>
  </tbody>
</table>
</div>



The optimal hidden layer size is 1 hidden layer with 200 neurons. We need a l2-regularization term with value 3 to achieve the best accuracy.

The overall test accuracy of neural networks is better than that of random forest classifier. It also has a significantly higher accuracy on `CN` cognitively normal people, while its accuracies on `CI` cognitive impairment and `AD` Alzheimer's disease are slightly lower. We would say that the neural networks model belongs to the group of Type III classifiers discussed in the previous section. It is also a very promising model.
