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
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
    if (not c.startswith('PT')) or (c=='PTEDUCAT')]

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


    (621, 74)





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
      <th>...</th>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
<p>5 rows × 74 columns</p>
</div>





```python
cv_fold = KFold(n_splits=4, shuffle=True, random_state=9001)
parameters = {'alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
              'hidden_layer_sizes': [(10), (25), (50), (100),
                                     (10, 10), (10, 25), (10, 50), 
                                     (25, 10), (25, 25), 
                                     (50, 10)]}
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
    L2 penalty parameter:  10.0
    Hidden Layer Sizes:  25
    
    -----------------
    
    Training accuracy:  0.824476650564
    Test accuracy:  0.777777777778
    
    -----------------
    
    Test Confusion Matrix: 
    [[24 18  0]
     [ 8 79  6]
     [ 0  4 23]]




```python
rf_best = RandomForestClassifier(n_estimators=16, max_depth=8, random_state=9001)
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
    
    Training accuracy:  0.972624798712
    Test accuracy:  0.796296296296
    
    -----------------
    
    Test Confusion Matrix: 
    [[20 22  0]
     [ 4 87  2]
     [ 0  5 22]]




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
      <td>0.824477</td>
      <td>0.972625</td>
    </tr>
    <tr>
      <th>Test accuracy</th>
      <td>0.777778</td>
      <td>0.796296</td>
    </tr>
    <tr>
      <th>Test accuracy CN</th>
      <td>0.571429</td>
      <td>0.476190</td>
    </tr>
    <tr>
      <th>Test accuracy CI</th>
      <td>0.849462</td>
      <td>0.935484</td>
    </tr>
    <tr>
      <th>Test accuracy AD</th>
      <td>0.851852</td>
      <td>0.814815</td>
    </tr>
  </tbody>
</table>
</div>



The optimal hidden layer size is one hidden layer with 25 neurons. We need a l2-regularization term with value 10 to achieve the best accuracy.

The overall test accuracy of neural networks is close to that of the best random forest in the previous model comparison section. However, we found that the test accuracy for the cognitively normal group and the Alzheimer's disease group is considerably higher using neural network. These two groups are what we want to differentiate.

Hence, we would prefer neural networks model to the random forest classifier.
