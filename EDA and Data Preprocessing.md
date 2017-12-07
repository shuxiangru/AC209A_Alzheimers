---
title: EDA and Data Preprocessing
notebook: EDA and Data Preprocessing.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.8)
```


## Load Data



```python
adnimerge_unique = pd.read_csv('data/ADNIMERGE_unique.csv')
print(adnimerge_unique.shape)
adnimerge_unique.head()
```


    (783, 89)





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
      <th>RID</th>
      <th>DX_bl</th>
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
      <th>APOE4_slope</th>
      <th>CSF_ABETA</th>
      <th>CSF_ABETA_slope</th>
      <th>CSF_TAU</th>
      <th>CSF_TAU_slope</th>
      <th>CSF_PTAU</th>
      <th>CSF_PTAU_slope</th>
      <th>FDG</th>
      <th>FDG_slope</th>
      <th>AV45</th>
      <th>AV45_slope</th>
      <th>CDRSB</th>
      <th>CDRSB_slope</th>
      <th>ADAS11</th>
      <th>ADAS11_slope</th>
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
      <th>EcogPtTotal</th>
      <th>EcogPtTotal_slope</th>
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
      <th>EcogSPTotal</th>
      <th>EcogSPTotal_slope</th>
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
      <td>4001</td>
      <td>2</td>
      <td>88.5</td>
      <td>0</td>
      <td>9</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>105.7</td>
      <td>0.000000e+00</td>
      <td>141.6</td>
      <td>3.006505e-15</td>
      <td>36.2</td>
      <td>1.503252e-15</td>
      <td>1.11537</td>
      <td>0.000033</td>
      <td>1.507200</td>
      <td>NaN</td>
      <td>5.5</td>
      <td>0.322375</td>
      <td>24.0</td>
      <td>0.249574</td>
      <td>38.0</td>
      <td>0.234855</td>
      <td>20.0</td>
      <td>-0.186659</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.500</td>
      <td>0.053791</td>
      <td>1.11111</td>
      <td>NaN</td>
      <td>1.00000</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.00000</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>1.14286</td>
      <td>NaN</td>
      <td>3.875</td>
      <td>0.004211</td>
      <td>1.00000</td>
      <td>0.039147</td>
      <td>2.25000</td>
      <td>0.043828</td>
      <td>3.6</td>
      <td>0.013474</td>
      <td>3.66667</td>
      <td>0.011228</td>
      <td>4.00</td>
      <td>-0.010675</td>
      <td>2.91667</td>
      <td>0.018605</td>
      <td>22.0</td>
      <td>0.147061</td>
      <td>33609.0</td>
      <td>327.921234</td>
      <td>5532.0</td>
      <td>-13.555558</td>
      <td>864483.0</td>
      <td>-1766.049081</td>
      <td>2995.0</td>
      <td>-64.791139</td>
      <td>14530.0</td>
      <td>-3.466522</td>
      <td>14249.0</td>
      <td>-47.827089</td>
      <td>1255450.0</td>
      <td>-0.543778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4004</td>
      <td>1</td>
      <td>66.8</td>
      <td>0</td>
      <td>14</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>208.1</td>
      <td>5.096268e-15</td>
      <td>83.3</td>
      <td>1.274067e-15</td>
      <td>33.9</td>
      <td>9.555503e-16</td>
      <td>1.26220</td>
      <td>-0.003933</td>
      <td>0.973711</td>
      <td>-0.000457</td>
      <td>1.5</td>
      <td>-0.021869</td>
      <td>4.0</td>
      <td>0.014977</td>
      <td>5.0</td>
      <td>0.051461</td>
      <td>30.0</td>
      <td>0.002234</td>
      <td>53.0</td>
      <td>-0.036797</td>
      <td>5.0</td>
      <td>0.033328</td>
      <td>9.0</td>
      <td>-0.070851</td>
      <td>69.2308</td>
      <td>-0.499578</td>
      <td>27.0</td>
      <td>0.047281</td>
      <td>1.750</td>
      <td>0.006296</td>
      <td>1.66667</td>
      <td>0.012584</td>
      <td>1.85714</td>
      <td>2.035398e-02</td>
      <td>1.4</td>
      <td>0.020378</td>
      <td>1.50000</td>
      <td>0.007639</td>
      <td>2.00</td>
      <td>0.009371</td>
      <td>1.69231</td>
      <td>0.012572</td>
      <td>1.625</td>
      <td>-0.005569</td>
      <td>1.22222</td>
      <td>0.000527</td>
      <td>1.28571</td>
      <td>-0.002936</td>
      <td>1.8</td>
      <td>-0.013355</td>
      <td>1.33333</td>
      <td>-0.005150</td>
      <td>2.25</td>
      <td>-0.016481</td>
      <td>1.51282</td>
      <td>-0.005680</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4005</td>
      <td>1</td>
      <td>70.5</td>
      <td>1</td>
      <td>16</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>139.4</td>
      <td>0.000000e+00</td>
      <td>128.0</td>
      <td>0.000000e+00</td>
      <td>73.9</td>
      <td>9.049212e-16</td>
      <td>1.25009</td>
      <td>0.000314</td>
      <td>1.395770</td>
      <td>0.001351</td>
      <td>3.5</td>
      <td>-0.044936</td>
      <td>6.0</td>
      <td>0.127031</td>
      <td>8.0</td>
      <td>0.155971</td>
      <td>29.0</td>
      <td>0.045185</td>
      <td>26.0</td>
      <td>0.189440</td>
      <td>1.0</td>
      <td>-0.024890</td>
      <td>6.0</td>
      <td>0.011707</td>
      <td>100.0000</td>
      <td>0.023501</td>
      <td>28.0</td>
      <td>0.003579</td>
      <td>2.125</td>
      <td>0.007081</td>
      <td>1.00000</td>
      <td>0.010400</td>
      <td>1.14286</td>
      <td>8.180210e-03</td>
      <td>1.4</td>
      <td>0.004251</td>
      <td>1.33333</td>
      <td>-0.005677</td>
      <td>1.50</td>
      <td>0.003515</td>
      <td>1.41026</td>
      <td>0.005353</td>
      <td>3.000</td>
      <td>0.013977</td>
      <td>2.33333</td>
      <td>0.009818</td>
      <td>2.57143</td>
      <td>0.026097</td>
      <td>2.2</td>
      <td>0.032259</td>
      <td>2.66667</td>
      <td>0.018311</td>
      <td>2.25</td>
      <td>0.023470</td>
      <td>2.53846</td>
      <td>0.019137</td>
      <td>8.0</td>
      <td>-0.061578</td>
      <td>38294.0</td>
      <td>268.520926</td>
      <td>7207.0</td>
      <td>-3.094318</td>
      <td>1181170.0</td>
      <td>-945.902809</td>
      <td>4405.0</td>
      <td>-1.281092</td>
      <td>22968.0</td>
      <td>-44.868160</td>
      <td>22654.0</td>
      <td>-40.787854</td>
      <td>1768220.0</td>
      <td>-346.827114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4007</td>
      <td>1</td>
      <td>78.1</td>
      <td>1</td>
      <td>20</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>144.3</td>
      <td>4.488822e-15</td>
      <td>86.2</td>
      <td>-2.244411e-15</td>
      <td>40.5</td>
      <td>-1.122205e-15</td>
      <td>1.33645</td>
      <td>0.003553</td>
      <td>1.653660</td>
      <td>0.002966</td>
      <td>0.5</td>
      <td>0.031164</td>
      <td>9.0</td>
      <td>-0.182366</td>
      <td>14.0</td>
      <td>-0.031164</td>
      <td>29.0</td>
      <td>0.046588</td>
      <td>40.0</td>
      <td>-0.174286</td>
      <td>7.0</td>
      <td>-0.159071</td>
      <td>4.0</td>
      <td>0.275438</td>
      <td>36.3636</td>
      <td>2.915744</td>
      <td>24.0</td>
      <td>-0.054248</td>
      <td>1.500</td>
      <td>-0.010637</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.14286</td>
      <td>2.273962e-17</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>1.12821</td>
      <td>-0.002182</td>
      <td>2.625</td>
      <td>-0.017011</td>
      <td>1.55556</td>
      <td>-0.026302</td>
      <td>1.83333</td>
      <td>-0.038269</td>
      <td>2.6</td>
      <td>-0.071456</td>
      <td>1.83333</td>
      <td>-0.038823</td>
      <td>2.75</td>
      <td>-0.036882</td>
      <td>2.13158</td>
      <td>-0.035686</td>
      <td>1.0</td>
      <td>-0.003882</td>
      <td>36679.0</td>
      <td>164.977666</td>
      <td>7495.0</td>
      <td>-25.103614</td>
      <td>1029740.0</td>
      <td>-865.813171</td>
      <td>3522.0</td>
      <td>0.192841</td>
      <td>19848.0</td>
      <td>-49.114981</td>
      <td>19938.0</td>
      <td>5.985021</td>
      <td>1426170.0</td>
      <td>-497.941640</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4009</td>
      <td>2</td>
      <td>90.3</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>233.9</td>
      <td>0.000000e+00</td>
      <td>71.4</td>
      <td>2.079238e-15</td>
      <td>22.1</td>
      <td>5.198096e-16</td>
      <td>1.17124</td>
      <td>-0.001039</td>
      <td>0.909650</td>
      <td>-0.002026</td>
      <td>8.0</td>
      <td>0.114822</td>
      <td>15.0</td>
      <td>0.070772</td>
      <td>25.0</td>
      <td>0.171000</td>
      <td>24.0</td>
      <td>0.073440</td>
      <td>25.0</td>
      <td>-0.222341</td>
      <td>1.0</td>
      <td>-0.043417</td>
      <td>3.0</td>
      <td>0.033877</td>
      <td>60.0000</td>
      <td>1.355067</td>
      <td>18.0</td>
      <td>-0.013394</td>
      <td>2.000</td>
      <td>-0.009109</td>
      <td>1.33333</td>
      <td>-0.009468</td>
      <td>2.14286</td>
      <td>-3.698373e-02</td>
      <td>1.8</td>
      <td>-0.017324</td>
      <td>2.00000</td>
      <td>-0.027624</td>
      <td>1.75</td>
      <td>-0.018569</td>
      <td>1.82051</td>
      <td>-0.019639</td>
      <td>3.625</td>
      <td>0.009662</td>
      <td>2.44444</td>
      <td>0.017177</td>
      <td>3.71429</td>
      <td>0.013120</td>
      <td>3.4</td>
      <td>0.003900</td>
      <td>3.00000</td>
      <td>0.032227</td>
      <td>2.75</td>
      <td>0.043571</td>
      <td>3.15385</td>
      <td>0.014850</td>
      <td>26.0</td>
      <td>0.235116</td>
      <td>29136.0</td>
      <td>8.365721</td>
      <td>5224.0</td>
      <td>-28.069806</td>
      <td>910905.0</td>
      <td>-3162.821533</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1338420.0</td>
      <td>-1049.248536</td>
    </tr>
  </tbody>
</table>
</div>





```python
cols_id = ['RID']
cols_predictand = ['DX_bl']
cols_demo_cont = ['PTAGE', 'PTEDUCAT']
cols_demo_cata = [c for c in adnimerge_unique.columns if c.startswith('PT') and c not in cols_demo_cont]
cols_longit_slope = [c for c in adnimerge_unique.columns if c.endswith('slope')]
cols_longit_bl = [c.replace('_slope','') for c in cols_longit_slope]
```




```python
num_all = adnimerge_unique.shape[0]
num_CN = np.sum(adnimerge_unique['DX_bl']==0)
num_AD = np.sum(adnimerge_unique['DX_bl']==2)
num_other = num_all - num_CN - num_AD

print('Number of patients in each category after modeling longitudinality:')
print('-------------------------------------------------------------------')
print('total:', num_all)
print('CN    patients:', num_CN)
print('AD    patients:', num_AD)
print('other patients:', num_other)
```


    Number of patients in each category after modeling longitudinality:
    -------------------------------------------------------------------
    total: 783
    CN    patients: 187
    AD    patients: 149
    other patients: 447


## Histogram

We plotted histograms for each predictor to look at the distribution of each category. To better classify Alzheimer's Disease, we would like to know the relationship of each variable and the diagnosis. In addition, we would like to know how much data is missing in each category of diagnosis and find the best imputing method based on the result.

Here are some noteworthy findings:
- `CDRSB`: All the cognitively normal `CN` patients have a value close to 0. As the value increases, it is more likely that the patient has Alzheimer’s disease. Cognitive impairment and Alzheimer’s Disease `AD` patients can have different values for these two variables. That shows if the value is not 0, then the patient is experiencing some sort of dementia. This could be a very strong predictor. However, `CDRSB` is actively used to deduce DX and will erroneously inflate accuracy. We decided to delete this variable along with its slope.
- `ADAS13`, `CSF_TAU` and `EcogSPMem`: From the plot, it seems that as the value increases, the likelihood of `AD` also increases. It indicates high values are associated with Alzheimer’s Disease. Three categories are well separated.
- `RAVLT_immediate` and `Hippocampus` have negative relationship with Alzheimer’s. As the value of these variables decreases, it is more likely to get AD.
- The missing rate is very low overall. For the predictors that have missing values, the precentage of missing value is similar across each category.

Note: The percentage value in the legend indicates the percentage of data that is not missing in each category



```python
num_cont = 1 + len(cols_longit_bl)

adnimerge_unique_CN = adnimerge_unique[adnimerge_unique['DX_bl']==0]
adnimerge_unique_MC = adnimerge_unique[adnimerge_unique['DX_bl']==1]
adnimerge_unique_AD = adnimerge_unique[adnimerge_unique['DX_bl']==2]

fig, ax = plt.subplots(num_cont, 2, figsize=(12,200))

for i,col_name in enumerate(cols_demo_cont):
    mis_rate_CN = np.mean(adnimerge_unique_CN[col_name].isnull())
    mis_rate_MC = np.mean(adnimerge_unique_MC[col_name].isnull())
    mis_rate_AD = np.mean(adnimerge_unique_AD[col_name].isnull())
    adnimerge_unique_CN[col_name].hist(
        ax=ax[0,i], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('CN', 100*(1-mis_rate_CN)))
    adnimerge_unique_MC[col_name].hist(
        ax=ax[0,i], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('CI', 100*(1-mis_rate_MC)))
    adnimerge_unique_AD[col_name].hist(
        ax=ax[0,i], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('AD', 100*(1-mis_rate_AD)))
    ax[0,i].set_title(col_name)
    ax[0,i].legend(loc='best')

for i,col_name in enumerate(cols_longit_bl):
    mis_rate_CN = np.mean(adnimerge_unique_CN[col_name].isnull())
    mis_rate_MC = np.mean(adnimerge_unique_MC[col_name].isnull())
    mis_rate_AD = np.mean(adnimerge_unique_AD[col_name].isnull())
    adnimerge_unique_CN[col_name].hist(
        ax=ax[i+1,0], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('CN', 100*(1-mis_rate_CN)))
    adnimerge_unique_MC[col_name].hist(
        ax=ax[i+1,0], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('CI', 100*(1-mis_rate_MC)))
    adnimerge_unique_AD[col_name].hist(
        ax=ax[i+1,0], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('AD', 100*(1-mis_rate_AD)))
    ax[i+1,0].set_title(col_name)
    ax[i+1,0].legend(loc='best')
    
for i,col_name in enumerate(cols_longit_slope):
    mis_rate_CN = np.mean(adnimerge_unique_CN[col_name].isnull())
    mis_rate_MC = np.mean(adnimerge_unique_MC[col_name].isnull())
    mis_rate_AD = np.mean(adnimerge_unique_AD[col_name].isnull())
    adnimerge_unique_CN[col_name].hist(
        ax=ax[i+1,1], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('CN', 100*(1-mis_rate_CN)))
    adnimerge_unique_MC[col_name].hist(
        ax=ax[i+1,1], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('CI', 100*(1-mis_rate_MC)))
    adnimerge_unique_AD[col_name].hist(
        ax=ax[i+1,1], alpha=0.4, 
        label='{:^4} ({:0.0f}%)'.format('AD', 100*(1-mis_rate_AD)))
    ax[i+1,1].set_title(col_name)
    ax[i+1,1].legend(loc='best')
```



![png](EDA%20and%20Data%20Preprocessing_files/EDA%20and%20Data%20Preprocessing_7_0.png)


## Correlation Heatmap

To avoid collinearity, we would like to examine the correlation between variables. We used heatmap to do so.

According to the heap map of all selected predictors, there are high correlations between the following
pairs of predictors: `CDRSB` vs `FAQ`, `ADAS11` vs `ADAS13`, `WholeBrain` vs `ICV`, and among all
`EcogXXX` predictors. To deal with these correlations, we deleted `ADAS11`, `EcogPtTotal` and `EcogSPTotal` based on the review of scientific papers that investigate the importance of the factors mentioned above related to the prediction of Alzheimer’s Disease. 

Note: The heatmap is obtained by only considering the value that are not None in both predictors.



```python
def plot_correlation_heatmap(df1, df2, ax):
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    N1 = df1.shape[1]
    N2 = df2.shape[1]
    corr_matrix = np.zeros((N1, N2))
    for i,icol in enumerate(df1.columns):
        for j,jcol in enumerate(df2.columns):
            idx_available = np.where(
                (~df1[icol].isnull()) & (~df2[jcol].isnull()))
            ival = df1[icol].values[idx_available]
            jval = df2[jcol].values[idx_available]
            cov = np.mean(ival*jval) - np.mean(ival)*np.mean(jval)
            corr = cov / np.std(ival) / np.std(jval)
            corr_matrix[i,j] = corr
            
    ax.set_xticks(ticks=range(len(df2.columns)))
    ax.set_yticks(ticks=range(len(df1.columns)))
    ax.set_xticklabels(df2.columns, rotation='vertical')
    ax.set_yticklabels(df1.columns)
    plot = ax.pcolor(np.abs(corr_matrix), vmin=0, vmax=1)
    return plot
```




```python
fig, ax = plt.subplots(figsize=(15,12))
plot = plot_correlation_heatmap(
    adnimerge_unique[cols_demo_cont+cols_longit_bl], 
    adnimerge_unique[cols_demo_cont+cols_longit_bl], ax)
fig.colorbar(plot)
```





    <matplotlib.colorbar.Colorbar at 0x1a1cd7c588>




![png](EDA%20and%20Data%20Preprocessing_files/EDA%20and%20Data%20Preprocessing_11_1.png)


We listed the predictors that have a small slope in magnitude related to its value. We deleted these slope variables, `APOE4_slope`, `CSF_ABETA_slope`, `CSF_TAU_slope`, and `CSF_PTAU_slope` as they could not show any trend of the following visits of the patients and were meaningless. We also checked the collinearity of the slopes themselves. We ended up with the same conclusion as above. To deal with these correlations, we deleted `ADAS11_slope`, `EcogPtTotal_slope` and `EcogSPTotal_slope`. Further, we found that the slopes are not collinear with the predictors. 



```python
tol_err = 1e-6
cols_longit_slope_valid = []
for cbl,csl in zip(cols_longit_bl,cols_longit_slope):
    bl = adnimerge_unique[cbl]
    sl = adnimerge_unique[csl]
    if np.std(sl) > tol_err*np.mean(bl):
        cols_longit_slope_valid.append(csl)
cols_longit_slope_invalid = [
    c for c in cols_longit_slope if c not in cols_longit_slope_valid]
print('Predictor slopes that are too small:\n', cols_longit_slope_invalid)
```


    Predictor slopes that are too small:
     ['APOE4_slope', 'CSF_ABETA_slope', 'CSF_TAU_slope', 'CSF_PTAU_slope']




```python
fig, ax = plt.subplots(figsize=(15,12))
plot = plot_correlation_heatmap(
    adnimerge_unique[cols_longit_slope_valid], 
    adnimerge_unique[cols_longit_slope_valid], ax)
fig.colorbar(plot)
```





    <matplotlib.colorbar.Colorbar at 0x1a1614fa20>




![png](EDA%20and%20Data%20Preprocessing_files/EDA%20and%20Data%20Preprocessing_14_1.png)




```python
fig, ax = plt.subplots(figsize=(15,12))
plot = plot_correlation_heatmap(
    adnimerge_unique[cols_longit_slope[4:]],
    adnimerge_unique[cols_demo_cont+cols_longit_bl], ax)
fig.colorbar(plot)
```





    <matplotlib.colorbar.Colorbar at 0x1a142a7550>




![png](EDA%20and%20Data%20Preprocessing_files/EDA%20and%20Data%20Preprocessing_15_1.png)


## Drop Predictors



```python
del adnimerge_unique['CDRSB']
del adnimerge_unique['CDRSB_slope']

del adnimerge_unique['ADAS11']
del adnimerge_unique['EcogPtTotal']
del adnimerge_unique['EcogSPTotal']
del adnimerge_unique['ADAS11_slope']
del adnimerge_unique['EcogPtTotal_slope']
del adnimerge_unique['EcogSPTotal_slope']

del adnimerge_unique['APOE4_slope']
del adnimerge_unique['CSF_ABETA_slope']
del adnimerge_unique['CSF_TAU_slope']
del adnimerge_unique['CSF_PTAU_slope']
```




```python
print(adnimerge_unique.shape)
adnimerge_unique.head()
```


    (783, 77)





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
      <th>RID</th>
      <th>DX_bl</th>
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
      <td>4001</td>
      <td>2</td>
      <td>88.5</td>
      <td>0</td>
      <td>9</td>
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
      <td>0.0</td>
      <td>105.7</td>
      <td>141.6</td>
      <td>36.2</td>
      <td>1.11537</td>
      <td>0.000033</td>
      <td>1.507200</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>0.234855</td>
      <td>20.0</td>
      <td>-0.186659</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.500</td>
      <td>0.053791</td>
      <td>1.11111</td>
      <td>NaN</td>
      <td>1.00000</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.00000</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>3.875</td>
      <td>0.004211</td>
      <td>1.00000</td>
      <td>0.039147</td>
      <td>2.25000</td>
      <td>0.043828</td>
      <td>3.6</td>
      <td>0.013474</td>
      <td>3.66667</td>
      <td>0.011228</td>
      <td>4.00</td>
      <td>-0.010675</td>
      <td>22.0</td>
      <td>0.147061</td>
      <td>33609.0</td>
      <td>327.921234</td>
      <td>5532.0</td>
      <td>-13.555558</td>
      <td>864483.0</td>
      <td>-1766.049081</td>
      <td>2995.0</td>
      <td>-64.791139</td>
      <td>14530.0</td>
      <td>-3.466522</td>
      <td>14249.0</td>
      <td>-47.827089</td>
      <td>1255450.0</td>
      <td>-0.543778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4004</td>
      <td>1</td>
      <td>66.8</td>
      <td>0</td>
      <td>14</td>
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
      <td>0.0</td>
      <td>208.1</td>
      <td>83.3</td>
      <td>33.9</td>
      <td>1.26220</td>
      <td>-0.003933</td>
      <td>0.973711</td>
      <td>-0.000457</td>
      <td>5.0</td>
      <td>0.051461</td>
      <td>30.0</td>
      <td>0.002234</td>
      <td>53.0</td>
      <td>-0.036797</td>
      <td>5.0</td>
      <td>0.033328</td>
      <td>9.0</td>
      <td>-0.070851</td>
      <td>69.2308</td>
      <td>-0.499578</td>
      <td>27.0</td>
      <td>0.047281</td>
      <td>1.750</td>
      <td>0.006296</td>
      <td>1.66667</td>
      <td>0.012584</td>
      <td>1.85714</td>
      <td>2.035398e-02</td>
      <td>1.4</td>
      <td>0.020378</td>
      <td>1.50000</td>
      <td>0.007639</td>
      <td>2.00</td>
      <td>0.009371</td>
      <td>1.625</td>
      <td>-0.005569</td>
      <td>1.22222</td>
      <td>0.000527</td>
      <td>1.28571</td>
      <td>-0.002936</td>
      <td>1.8</td>
      <td>-0.013355</td>
      <td>1.33333</td>
      <td>-0.005150</td>
      <td>2.25</td>
      <td>-0.016481</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4005</td>
      <td>1</td>
      <td>70.5</td>
      <td>1</td>
      <td>16</td>
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
      <td>1.0</td>
      <td>139.4</td>
      <td>128.0</td>
      <td>73.9</td>
      <td>1.25009</td>
      <td>0.000314</td>
      <td>1.395770</td>
      <td>0.001351</td>
      <td>8.0</td>
      <td>0.155971</td>
      <td>29.0</td>
      <td>0.045185</td>
      <td>26.0</td>
      <td>0.189440</td>
      <td>1.0</td>
      <td>-0.024890</td>
      <td>6.0</td>
      <td>0.011707</td>
      <td>100.0000</td>
      <td>0.023501</td>
      <td>28.0</td>
      <td>0.003579</td>
      <td>2.125</td>
      <td>0.007081</td>
      <td>1.00000</td>
      <td>0.010400</td>
      <td>1.14286</td>
      <td>8.180210e-03</td>
      <td>1.4</td>
      <td>0.004251</td>
      <td>1.33333</td>
      <td>-0.005677</td>
      <td>1.50</td>
      <td>0.003515</td>
      <td>3.000</td>
      <td>0.013977</td>
      <td>2.33333</td>
      <td>0.009818</td>
      <td>2.57143</td>
      <td>0.026097</td>
      <td>2.2</td>
      <td>0.032259</td>
      <td>2.66667</td>
      <td>0.018311</td>
      <td>2.25</td>
      <td>0.023470</td>
      <td>8.0</td>
      <td>-0.061578</td>
      <td>38294.0</td>
      <td>268.520926</td>
      <td>7207.0</td>
      <td>-3.094318</td>
      <td>1181170.0</td>
      <td>-945.902809</td>
      <td>4405.0</td>
      <td>-1.281092</td>
      <td>22968.0</td>
      <td>-44.868160</td>
      <td>22654.0</td>
      <td>-40.787854</td>
      <td>1768220.0</td>
      <td>-346.827114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4007</td>
      <td>1</td>
      <td>78.1</td>
      <td>1</td>
      <td>20</td>
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
      <td>1.0</td>
      <td>144.3</td>
      <td>86.2</td>
      <td>40.5</td>
      <td>1.33645</td>
      <td>0.003553</td>
      <td>1.653660</td>
      <td>0.002966</td>
      <td>14.0</td>
      <td>-0.031164</td>
      <td>29.0</td>
      <td>0.046588</td>
      <td>40.0</td>
      <td>-0.174286</td>
      <td>7.0</td>
      <td>-0.159071</td>
      <td>4.0</td>
      <td>0.275438</td>
      <td>36.3636</td>
      <td>2.915744</td>
      <td>24.0</td>
      <td>-0.054248</td>
      <td>1.500</td>
      <td>-0.010637</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.14286</td>
      <td>2.273962e-17</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>2.625</td>
      <td>-0.017011</td>
      <td>1.55556</td>
      <td>-0.026302</td>
      <td>1.83333</td>
      <td>-0.038269</td>
      <td>2.6</td>
      <td>-0.071456</td>
      <td>1.83333</td>
      <td>-0.038823</td>
      <td>2.75</td>
      <td>-0.036882</td>
      <td>1.0</td>
      <td>-0.003882</td>
      <td>36679.0</td>
      <td>164.977666</td>
      <td>7495.0</td>
      <td>-25.103614</td>
      <td>1029740.0</td>
      <td>-865.813171</td>
      <td>3522.0</td>
      <td>0.192841</td>
      <td>19848.0</td>
      <td>-49.114981</td>
      <td>19938.0</td>
      <td>5.985021</td>
      <td>1426170.0</td>
      <td>-497.941640</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4009</td>
      <td>2</td>
      <td>90.3</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>233.9</td>
      <td>71.4</td>
      <td>22.1</td>
      <td>1.17124</td>
      <td>-0.001039</td>
      <td>0.909650</td>
      <td>-0.002026</td>
      <td>25.0</td>
      <td>0.171000</td>
      <td>24.0</td>
      <td>0.073440</td>
      <td>25.0</td>
      <td>-0.222341</td>
      <td>1.0</td>
      <td>-0.043417</td>
      <td>3.0</td>
      <td>0.033877</td>
      <td>60.0000</td>
      <td>1.355067</td>
      <td>18.0</td>
      <td>-0.013394</td>
      <td>2.000</td>
      <td>-0.009109</td>
      <td>1.33333</td>
      <td>-0.009468</td>
      <td>2.14286</td>
      <td>-3.698373e-02</td>
      <td>1.8</td>
      <td>-0.017324</td>
      <td>2.00000</td>
      <td>-0.027624</td>
      <td>1.75</td>
      <td>-0.018569</td>
      <td>3.625</td>
      <td>0.009662</td>
      <td>2.44444</td>
      <td>0.017177</td>
      <td>3.71429</td>
      <td>0.013120</td>
      <td>3.4</td>
      <td>0.003900</td>
      <td>3.00000</td>
      <td>0.032227</td>
      <td>2.75</td>
      <td>0.043571</td>
      <td>26.0</td>
      <td>0.235116</td>
      <td>29136.0</td>
      <td>8.365721</td>
      <td>5224.0</td>
      <td>-28.069806</td>
      <td>910905.0</td>
      <td>-3162.821533</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1338420.0</td>
      <td>-1049.248536</td>
    </tr>
  </tbody>
</table>
</div>



## Imputation

The problem of missing values lies in the continuous variables. We imputed the missing values by mean. We did not impute by regression models because there were very few columns that had no missing data, and we could not build a satisfactory regression model based on that.



```python
def imputation_mean(df, cols):
    df_copy = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        c_mean = np.mean(df[c])
        df_copy[c] = df[c].map(lambda x:c_mean if np.isnan(x) else x)
    return df_copy
```




```python
cols_to_impute = cols_longit_bl + cols_longit_slope_valid
adnimerge_unique_imputed = imputation_mean(adnimerge_unique, cols_to_impute)
adnimerge_unique_imputed.head()
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
      <th>RID</th>
      <th>DX_bl</th>
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
      <td>4001</td>
      <td>2</td>
      <td>88.5</td>
      <td>0</td>
      <td>9</td>
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
      <td>0.0</td>
      <td>105.7</td>
      <td>141.6</td>
      <td>36.2</td>
      <td>1.11537</td>
      <td>0.000033</td>
      <td>1.507200</td>
      <td>0.000502</td>
      <td>38.0</td>
      <td>0.234855</td>
      <td>20.0</td>
      <td>-0.186659</td>
      <td>37.18822</td>
      <td>-0.103010</td>
      <td>4.539052</td>
      <td>-0.013343</td>
      <td>4.368758</td>
      <td>-0.010724</td>
      <td>56.16988</td>
      <td>0.180313</td>
      <td>22.979301</td>
      <td>-0.046780</td>
      <td>1.500</td>
      <td>0.053791</td>
      <td>1.11111</td>
      <td>-0.000064</td>
      <td>1.00000</td>
      <td>2.868415e-03</td>
      <td>1.0</td>
      <td>0.002403</td>
      <td>1.00000</td>
      <td>0.002204</td>
      <td>1.00</td>
      <td>0.002802</td>
      <td>3.875</td>
      <td>0.004211</td>
      <td>1.00000</td>
      <td>0.039147</td>
      <td>2.25000</td>
      <td>0.043828</td>
      <td>3.6</td>
      <td>0.013474</td>
      <td>3.66667</td>
      <td>0.011228</td>
      <td>4.00</td>
      <td>-0.010675</td>
      <td>22.0</td>
      <td>0.147061</td>
      <td>33609.000000</td>
      <td>327.921234</td>
      <td>5532.00000</td>
      <td>-13.555558</td>
      <td>8.644830e+05</td>
      <td>-1766.049081</td>
      <td>2995.000000</td>
      <td>-64.791139</td>
      <td>14530.000000</td>
      <td>-3.466522</td>
      <td>14249.000000</td>
      <td>-47.827089</td>
      <td>1.255450e+06</td>
      <td>-0.543778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4004</td>
      <td>1</td>
      <td>66.8</td>
      <td>0</td>
      <td>14</td>
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
      <td>0.0</td>
      <td>208.1</td>
      <td>83.3</td>
      <td>33.9</td>
      <td>1.26220</td>
      <td>-0.003933</td>
      <td>0.973711</td>
      <td>-0.000457</td>
      <td>5.0</td>
      <td>0.051461</td>
      <td>30.0</td>
      <td>0.002234</td>
      <td>53.00000</td>
      <td>-0.036797</td>
      <td>5.000000</td>
      <td>0.033328</td>
      <td>9.000000</td>
      <td>-0.070851</td>
      <td>69.23080</td>
      <td>-0.499578</td>
      <td>27.000000</td>
      <td>0.047281</td>
      <td>1.750</td>
      <td>0.006296</td>
      <td>1.66667</td>
      <td>0.012584</td>
      <td>1.85714</td>
      <td>2.035398e-02</td>
      <td>1.4</td>
      <td>0.020378</td>
      <td>1.50000</td>
      <td>0.007639</td>
      <td>2.00</td>
      <td>0.009371</td>
      <td>1.625</td>
      <td>-0.005569</td>
      <td>1.22222</td>
      <td>0.000527</td>
      <td>1.28571</td>
      <td>-0.002936</td>
      <td>1.8</td>
      <td>-0.013355</td>
      <td>1.33333</td>
      <td>-0.005150</td>
      <td>2.25</td>
      <td>-0.016481</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>37648.184722</td>
      <td>221.307493</td>
      <td>6992.62446</td>
      <td>-13.605642</td>
      <td>1.043388e+06</td>
      <td>-824.231999</td>
      <td>3576.089971</td>
      <td>-8.352327</td>
      <td>18087.830383</td>
      <td>-32.266216</td>
      <td>19935.085546</td>
      <td>-32.674813</td>
      <td>1.502300e+06</td>
      <td>-124.516227</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4005</td>
      <td>1</td>
      <td>70.5</td>
      <td>1</td>
      <td>16</td>
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
      <td>1.0</td>
      <td>139.4</td>
      <td>128.0</td>
      <td>73.9</td>
      <td>1.25009</td>
      <td>0.000314</td>
      <td>1.395770</td>
      <td>0.001351</td>
      <td>8.0</td>
      <td>0.155971</td>
      <td>29.0</td>
      <td>0.045185</td>
      <td>26.00000</td>
      <td>0.189440</td>
      <td>1.000000</td>
      <td>-0.024890</td>
      <td>6.000000</td>
      <td>0.011707</td>
      <td>100.00000</td>
      <td>0.023501</td>
      <td>28.000000</td>
      <td>0.003579</td>
      <td>2.125</td>
      <td>0.007081</td>
      <td>1.00000</td>
      <td>0.010400</td>
      <td>1.14286</td>
      <td>8.180210e-03</td>
      <td>1.4</td>
      <td>0.004251</td>
      <td>1.33333</td>
      <td>-0.005677</td>
      <td>1.50</td>
      <td>0.003515</td>
      <td>3.000</td>
      <td>0.013977</td>
      <td>2.33333</td>
      <td>0.009818</td>
      <td>2.57143</td>
      <td>0.026097</td>
      <td>2.2</td>
      <td>0.032259</td>
      <td>2.66667</td>
      <td>0.018311</td>
      <td>2.25</td>
      <td>0.023470</td>
      <td>8.0</td>
      <td>-0.061578</td>
      <td>38294.000000</td>
      <td>268.520926</td>
      <td>7207.00000</td>
      <td>-3.094318</td>
      <td>1.181170e+06</td>
      <td>-945.902809</td>
      <td>4405.000000</td>
      <td>-1.281092</td>
      <td>22968.000000</td>
      <td>-44.868160</td>
      <td>22654.000000</td>
      <td>-40.787854</td>
      <td>1.768220e+06</td>
      <td>-346.827114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4007</td>
      <td>1</td>
      <td>78.1</td>
      <td>1</td>
      <td>20</td>
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
      <td>1.0</td>
      <td>144.3</td>
      <td>86.2</td>
      <td>40.5</td>
      <td>1.33645</td>
      <td>0.003553</td>
      <td>1.653660</td>
      <td>0.002966</td>
      <td>14.0</td>
      <td>-0.031164</td>
      <td>29.0</td>
      <td>0.046588</td>
      <td>40.00000</td>
      <td>-0.174286</td>
      <td>7.000000</td>
      <td>-0.159071</td>
      <td>4.000000</td>
      <td>0.275438</td>
      <td>36.36360</td>
      <td>2.915744</td>
      <td>24.000000</td>
      <td>-0.054248</td>
      <td>1.500</td>
      <td>-0.010637</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.14286</td>
      <td>2.273962e-17</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>2.625</td>
      <td>-0.017011</td>
      <td>1.55556</td>
      <td>-0.026302</td>
      <td>1.83333</td>
      <td>-0.038269</td>
      <td>2.6</td>
      <td>-0.071456</td>
      <td>1.83333</td>
      <td>-0.038823</td>
      <td>2.75</td>
      <td>-0.036882</td>
      <td>1.0</td>
      <td>-0.003882</td>
      <td>36679.000000</td>
      <td>164.977666</td>
      <td>7495.00000</td>
      <td>-25.103614</td>
      <td>1.029740e+06</td>
      <td>-865.813171</td>
      <td>3522.000000</td>
      <td>0.192841</td>
      <td>19848.000000</td>
      <td>-49.114981</td>
      <td>19938.000000</td>
      <td>5.985021</td>
      <td>1.426170e+06</td>
      <td>-497.941640</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4009</td>
      <td>2</td>
      <td>90.3</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>233.9</td>
      <td>71.4</td>
      <td>22.1</td>
      <td>1.17124</td>
      <td>-0.001039</td>
      <td>0.909650</td>
      <td>-0.002026</td>
      <td>25.0</td>
      <td>0.171000</td>
      <td>24.0</td>
      <td>0.073440</td>
      <td>25.00000</td>
      <td>-0.222341</td>
      <td>1.000000</td>
      <td>-0.043417</td>
      <td>3.000000</td>
      <td>0.033877</td>
      <td>60.00000</td>
      <td>1.355067</td>
      <td>18.000000</td>
      <td>-0.013394</td>
      <td>2.000</td>
      <td>-0.009109</td>
      <td>1.33333</td>
      <td>-0.009468</td>
      <td>2.14286</td>
      <td>-3.698373e-02</td>
      <td>1.8</td>
      <td>-0.017324</td>
      <td>2.00000</td>
      <td>-0.027624</td>
      <td>1.75</td>
      <td>-0.018569</td>
      <td>3.625</td>
      <td>0.009662</td>
      <td>2.44444</td>
      <td>0.017177</td>
      <td>3.71429</td>
      <td>0.013120</td>
      <td>3.4</td>
      <td>0.003900</td>
      <td>3.00000</td>
      <td>0.032227</td>
      <td>2.75</td>
      <td>0.043571</td>
      <td>26.0</td>
      <td>0.235116</td>
      <td>29136.000000</td>
      <td>8.365721</td>
      <td>5224.00000</td>
      <td>-28.069806</td>
      <td>9.109050e+05</td>
      <td>-3162.821533</td>
      <td>3576.089971</td>
      <td>-8.352327</td>
      <td>18087.830383</td>
      <td>-32.266216</td>
      <td>19935.085546</td>
      <td>-32.674813</td>
      <td>1.338420e+06</td>
      <td>-1049.248536</td>
    </tr>
  </tbody>
</table>
</div>



## Train Test Split

We used 80% of the data as training set and the remaining 20% as the test set.



```python
np.random.seed(9001)
msk = np.random.rand(adnimerge_unique_imputed.shape[0]) < 0.8
df_train = adnimerge_unique_imputed[msk]
df_test = adnimerge_unique_imputed[~msk]
print(df_train.shape)
print(df_test.shape)
```


    (621, 77)
    (162, 77)




```python
df_train.to_csv('data/ADNIMERGE_train.csv', index=False)
df_test.to_csv('data/ADNIMERGE_test.csv', index=False)
```

