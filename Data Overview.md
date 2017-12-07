---
title: Data Overview
notebook: Data Overview.ipynb
nav_include: 2
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
```


## Merge

We analyzed the data collected under both ADNI1 and ADNI2 protocol and found that the missing rate was extremely high for many important predictors in ADNI1. Even though we have slightly more observations (819 patients) under ADNI1 protocol, ADNI2 data for patients (789 patients) is more complete.  Thus, we decided to base our model only on the data collected under ADNI2 protocol. 

Biomarkers may exist before clinical symptoms arise, and they could help us predict the onslaught of Alzheimer’s disease. To include more information about biomarkers, we merged the data in ADNIMERGE with UPenn CSF biomarkers. This file contains three biomarkers, `CSF_ABETA`, `CSF_TAU` and `CSF_PTAU`.



```python
adnimerge = pd.read_csv("data/ADNIMERGE.csv")
adnimerge = adnimerge[(adnimerge.COLPROT=="ADNI2")&(adnimerge.ORIGPROT=="ADNI2")]

cols_effective_biomarker = ['RID', 'ABETA', 'TAU', 'PTAU']
CSF_biomarker = pd.read_csv("data/UPENN_CSF Biomarkers_baseline_May15.2014.csv")
CSF_biomarker = CSF_biomarker[cols_effective_biomarker]
CSF_biomarker = CSF_biomarker.rename(columns=lambda s:'CSF_'+s)

adnimerge = adnimerge.merge(CSF_biomarker, how='left', left_on='RID', right_on='CSF_RID')

print(adnimerge.shape)
adnimerge.head()
```


    (4717, 98)





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
      <th>PTID</th>
      <th>VISCODE</th>
      <th>SITE</th>
      <th>COLPROT</th>
      <th>ORIGPROT</th>
      <th>EXAMDATE</th>
      <th>DX_bl</th>
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>PTETHCAT</th>
      <th>PTRACCAT</th>
      <th>PTMARRY</th>
      <th>APOE4</th>
      <th>FDG</th>
      <th>PIB</th>
      <th>AV45</th>
      <th>CDRSB</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>MMSE</th>
      <th>RAVLT_immediate</th>
      <th>RAVLT_learning</th>
      <th>RAVLT_forgetting</th>
      <th>RAVLT_perc_forgetting</th>
      <th>FAQ</th>
      <th>MOCA</th>
      <th>EcogPtMem</th>
      <th>EcogPtLang</th>
      <th>EcogPtVisspat</th>
      <th>EcogPtPlan</th>
      <th>EcogPtOrgan</th>
      <th>EcogPtDivatt</th>
      <th>EcogPtTotal</th>
      <th>EcogSPMem</th>
      <th>EcogSPLang</th>
      <th>EcogSPVisspat</th>
      <th>EcogSPPlan</th>
      <th>EcogSPOrgan</th>
      <th>EcogSPDivatt</th>
      <th>EcogSPTotal</th>
      <th>FLDSTRENG</th>
      <th>FSVERSION</th>
      <th>Ventricles</th>
      <th>Hippocampus</th>
      <th>WholeBrain</th>
      <th>Entorhinal</th>
      <th>Fusiform</th>
      <th>MidTemp</th>
      <th>ICV</th>
      <th>DX</th>
      <th>EXAMDATE_bl</th>
      <th>CDRSB_bl</th>
      <th>ADAS11_bl</th>
      <th>ADAS13_bl</th>
      <th>MMSE_bl</th>
      <th>RAVLT_immediate_bl</th>
      <th>RAVLT_learning_bl</th>
      <th>RAVLT_forgetting_bl</th>
      <th>RAVLT_perc_forgetting_bl</th>
      <th>FAQ_bl</th>
      <th>FLDSTRENG_bl</th>
      <th>FSVERSION_bl</th>
      <th>Ventricles_bl</th>
      <th>Hippocampus_bl</th>
      <th>WholeBrain_bl</th>
      <th>Entorhinal_bl</th>
      <th>Fusiform_bl</th>
      <th>MidTemp_bl</th>
      <th>ICV_bl</th>
      <th>MOCA_bl</th>
      <th>EcogPtMem_bl</th>
      <th>EcogPtLang_bl</th>
      <th>EcogPtVisspat_bl</th>
      <th>EcogPtPlan_bl</th>
      <th>EcogPtOrgan_bl</th>
      <th>EcogPtDivatt_bl</th>
      <th>EcogPtTotal_bl</th>
      <th>EcogSPMem_bl</th>
      <th>EcogSPLang_bl</th>
      <th>EcogSPVisspat_bl</th>
      <th>EcogSPPlan_bl</th>
      <th>EcogSPOrgan_bl</th>
      <th>EcogSPDivatt_bl</th>
      <th>EcogSPTotal_bl</th>
      <th>FDG_bl</th>
      <th>PIB_bl</th>
      <th>AV45_bl</th>
      <th>Years_bl</th>
      <th>Month_bl</th>
      <th>Month</th>
      <th>M</th>
      <th>update_stamp</th>
      <th>CSF_RID</th>
      <th>CSF_ABETA</th>
      <th>CSF_TAU</th>
      <th>CSF_PTAU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5282</td>
      <td>082_S_5282</td>
      <td>bl</td>
      <td>82</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>09/09/13</td>
      <td>SMC</td>
      <td>66.9</td>
      <td>Male</td>
      <td>17</td>
      <td>Not Hisp/Latino</td>
      <td>White</td>
      <td>Married</td>
      <td>1.0</td>
      <td>1.13549</td>
      <td>NaN</td>
      <td>1.326790</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>42.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>20.0000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.625</td>
      <td>1.33333</td>
      <td>1.14286</td>
      <td>1.2</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.25641</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>NaN</td>
      <td>7851.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1498720.0</td>
      <td>CN</td>
      <td>09/09/13</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>29</td>
      <td>42.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>NaN</td>
      <td>7851.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1498720.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.625</td>
      <td>1.33333</td>
      <td>1.14286</td>
      <td>1.2</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.25641</td>
      <td>1.13549</td>
      <td>NaN</td>
      <td>1.326790</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>20:00.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5280</td>
      <td>100_S_5280</td>
      <td>m24</td>
      <td>100</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>29/09/15</td>
      <td>SMC</td>
      <td>67.5</td>
      <td>Male</td>
      <td>16</td>
      <td>Not Hisp/Latino</td>
      <td>Black</td>
      <td>Never married</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.982566</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>50.0000</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.14286</td>
      <td>1.2</td>
      <td>1.66667</td>
      <td>1.50</td>
      <td>1.35897</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>CN</td>
      <td>17/09/13</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30</td>
      <td>42.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>33185.0</td>
      <td>8297.0</td>
      <td>1165500.0</td>
      <td>4946.0</td>
      <td>20147.0</td>
      <td>21194.0</td>
      <td>1656460.0</td>
      <td>28.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.00000</td>
      <td>2.00</td>
      <td>1.43590</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.25000</td>
      <td>1.00000</td>
      <td>1.25000</td>
      <td>1.25195</td>
      <td>NaN</td>
      <td>0.983143</td>
      <td>2.031490</td>
      <td>24.32790</td>
      <td>24</td>
      <td>24</td>
      <td>20:00.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5280</td>
      <td>100_S_5280</td>
      <td>m06</td>
      <td>100</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>06/03/14</td>
      <td>SMC</td>
      <td>67.5</td>
      <td>Male</td>
      <td>16</td>
      <td>Not Hisp/Latino</td>
      <td>Black</td>
      <td>Never married</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>29.0</td>
      <td>54.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>16.6667</td>
      <td>0.0</td>
      <td>24.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.2</td>
      <td>1.40000</td>
      <td>1.25</td>
      <td>1.28947</td>
      <td>1.750</td>
      <td>1.11111</td>
      <td>1.33333</td>
      <td>1.2</td>
      <td>NaN</td>
      <td>1.25000</td>
      <td>1.40000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CN</td>
      <td>17/09/13</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30</td>
      <td>42.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>33185.0</td>
      <td>8297.0</td>
      <td>1165500.0</td>
      <td>4946.0</td>
      <td>20147.0</td>
      <td>21194.0</td>
      <td>1656460.0</td>
      <td>28.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.00000</td>
      <td>2.00</td>
      <td>1.43590</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.25000</td>
      <td>1.00000</td>
      <td>1.25000</td>
      <td>1.25195</td>
      <td>NaN</td>
      <td>0.983143</td>
      <td>0.465435</td>
      <td>5.57377</td>
      <td>6</td>
      <td>6</td>
      <td>20:00.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5280</td>
      <td>100_S_5280</td>
      <td>bl</td>
      <td>100</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>17/09/13</td>
      <td>SMC</td>
      <td>67.5</td>
      <td>Male</td>
      <td>16</td>
      <td>Not Hisp/Latino</td>
      <td>Black</td>
      <td>Never married</td>
      <td>0.0</td>
      <td>1.25195</td>
      <td>NaN</td>
      <td>0.983143</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>42.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0000</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>2.00000</td>
      <td>2.00</td>
      <td>1.43590</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.25000</td>
      <td>1.00000</td>
      <td>1.25000</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>33185.0</td>
      <td>8297.0</td>
      <td>1165500.0</td>
      <td>4946.0</td>
      <td>20147.0</td>
      <td>21194.0</td>
      <td>1656460.0</td>
      <td>CN</td>
      <td>17/09/13</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30</td>
      <td>42.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>33185.0</td>
      <td>8297.0</td>
      <td>1165500.0</td>
      <td>4946.0</td>
      <td>20147.0</td>
      <td>21194.0</td>
      <td>1656460.0</td>
      <td>28.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.00000</td>
      <td>2.00</td>
      <td>1.43590</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.25000</td>
      <td>1.00000</td>
      <td>1.25000</td>
      <td>1.25195</td>
      <td>NaN</td>
      <td>0.983143</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>20:00.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5279</td>
      <td>082_S_5279</td>
      <td>bl</td>
      <td>82</td>
      <td>ADNI2</td>
      <td>ADNI2</td>
      <td>23/10/13</td>
      <td>SMC</td>
      <td>68.5</td>
      <td>Male</td>
      <td>20</td>
      <td>Not Hisp/Latino</td>
      <td>White</td>
      <td>Married</td>
      <td>0.0</td>
      <td>1.50629</td>
      <td>NaN</td>
      <td>0.985156</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>30.0</td>
      <td>61.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>20.0000</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>1.250</td>
      <td>1.44444</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>1.16667</td>
      <td>1.75</td>
      <td>1.25641</td>
      <td>1.875</td>
      <td>1.22222</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.33333</td>
      <td>1.33333</td>
      <td>1.37838</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>21327.0</td>
      <td>7654.0</td>
      <td>1081140.0</td>
      <td>4065.0</td>
      <td>17964.0</td>
      <td>18611.0</td>
      <td>1508210.0</td>
      <td>CN</td>
      <td>23/10/13</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>30</td>
      <td>61.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>3 Tesla MRI</td>
      <td>Cross-Sectional FreeSurfer (5.1)</td>
      <td>21327.0</td>
      <td>7654.0</td>
      <td>1081140.0</td>
      <td>4065.0</td>
      <td>17964.0</td>
      <td>18611.0</td>
      <td>1508210.0</td>
      <td>27.0</td>
      <td>1.250</td>
      <td>1.44444</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.16667</td>
      <td>1.75</td>
      <td>1.25641</td>
      <td>1.875</td>
      <td>1.22222</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.33333</td>
      <td>1.33333</td>
      <td>1.37838</td>
      <td>1.50629</td>
      <td>NaN</td>
      <td>0.985156</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>20:00.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>





```python
num_all = np.sum(adnimerge['VISCODE']=='bl')
num_CN = np.sum((adnimerge['DX_bl']=='CN')&(adnimerge['VISCODE']=='bl'))
num_AD = np.sum((adnimerge['DX_bl']=='AD')&(adnimerge['VISCODE']=='bl'))
num_other = num_all - num_CN - num_AD

print("Number of patients in each category")
print("-----------------------------------")
print('total:', num_all)
print('CN    patients:', num_CN)
print('AD    patients:', num_AD)
print('other patients:', num_other)
```


    Number of patients in each category
    -----------------------------------
    total: 789
    CN    patients: 188
    AD    patients: 150
    other patients: 451


## Pick Predictors

Deleted Columns & Reasons:
1. Unnecessary information due to data selection: `COLPROT`, `ORIGPROT`
2. Unrelated information: `PTID`, `SITE`,`EXAMDATE`,`FLDSTRENG`, `FSVERSION`
3. Information explained by or can be retrieved by other predictors: `VISCODE`, all `xxx_bl` predictors except `Month_bl`
4. Complete data missing due to group selection: `PIB`



```python
cols_of_interests = ['RID', 'DX_bl', 
                     'AGE', 'PTGENDER', 'PTEDUCAT', 
                     'PTETHCAT', 'PTRACCAT', 'PTMARRY', 
                     'APOE4', 'CSF_ABETA', 'CSF_TAU', 'CSF_PTAU', 
                     'FDG', 'AV45', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 
                     'RAVLT_immediate', 'RAVLT_learning', 
                     'RAVLT_forgetting', 'RAVLT_perc_forgetting', 
                     'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 
                     'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt', 
                     'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 
                     'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 
                     'EcogSPDivatt', 'EcogSPTotal', 'FAQ', 
                     'Ventricles', 'Hippocampus', 'WholeBrain', 
                     'Entorhinal', 'Fusiform', 'MidTemp', 
                     'ICV', 'DX', 'Month_bl', 'Month']

adnimerge_clean = adnimerge[cols_of_interests].copy()
print(adnimerge_clean.shape)
adnimerge_clean.head()
```


    (4717, 48)





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
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>PTETHCAT</th>
      <th>PTRACCAT</th>
      <th>PTMARRY</th>
      <th>APOE4</th>
      <th>CSF_ABETA</th>
      <th>CSF_TAU</th>
      <th>CSF_PTAU</th>
      <th>FDG</th>
      <th>AV45</th>
      <th>CDRSB</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>MMSE</th>
      <th>RAVLT_immediate</th>
      <th>RAVLT_learning</th>
      <th>RAVLT_forgetting</th>
      <th>RAVLT_perc_forgetting</th>
      <th>MOCA</th>
      <th>EcogPtMem</th>
      <th>EcogPtLang</th>
      <th>EcogPtVisspat</th>
      <th>EcogPtPlan</th>
      <th>EcogPtOrgan</th>
      <th>EcogPtDivatt</th>
      <th>EcogPtTotal</th>
      <th>EcogSPMem</th>
      <th>EcogSPLang</th>
      <th>EcogSPVisspat</th>
      <th>EcogSPPlan</th>
      <th>EcogSPOrgan</th>
      <th>EcogSPDivatt</th>
      <th>EcogSPTotal</th>
      <th>FAQ</th>
      <th>Ventricles</th>
      <th>Hippocampus</th>
      <th>WholeBrain</th>
      <th>Entorhinal</th>
      <th>Fusiform</th>
      <th>MidTemp</th>
      <th>ICV</th>
      <th>DX</th>
      <th>Month_bl</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5282</td>
      <td>SMC</td>
      <td>66.9</td>
      <td>Male</td>
      <td>17</td>
      <td>Not Hisp/Latino</td>
      <td>White</td>
      <td>Married</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.13549</td>
      <td>1.326790</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>42.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>20.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.625</td>
      <td>1.33333</td>
      <td>1.14286</td>
      <td>1.2</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.25641</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>7851.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1498720.0</td>
      <td>CN</td>
      <td>0.00000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5280</td>
      <td>SMC</td>
      <td>67.5</td>
      <td>Male</td>
      <td>16</td>
      <td>Not Hisp/Latino</td>
      <td>Black</td>
      <td>Never married</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.982566</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>50.0000</td>
      <td>27.0</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.14286</td>
      <td>1.2</td>
      <td>1.66667</td>
      <td>1.50</td>
      <td>1.35897</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CN</td>
      <td>24.32790</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5280</td>
      <td>SMC</td>
      <td>67.5</td>
      <td>Male</td>
      <td>16</td>
      <td>Not Hisp/Latino</td>
      <td>Black</td>
      <td>Never married</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>29.0</td>
      <td>54.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>16.6667</td>
      <td>24.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.2</td>
      <td>1.40000</td>
      <td>1.25</td>
      <td>1.28947</td>
      <td>1.750</td>
      <td>1.11111</td>
      <td>1.33333</td>
      <td>1.2</td>
      <td>NaN</td>
      <td>1.25000</td>
      <td>1.40000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CN</td>
      <td>5.57377</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5280</td>
      <td>SMC</td>
      <td>67.5</td>
      <td>Male</td>
      <td>16</td>
      <td>Not Hisp/Latino</td>
      <td>Black</td>
      <td>Never married</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.25195</td>
      <td>0.983143</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>42.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0000</td>
      <td>28.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>2.00000</td>
      <td>2.00</td>
      <td>1.43590</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.25000</td>
      <td>1.00000</td>
      <td>1.25000</td>
      <td>0.0</td>
      <td>33185.0</td>
      <td>8297.0</td>
      <td>1165500.0</td>
      <td>4946.0</td>
      <td>20147.0</td>
      <td>21194.0</td>
      <td>1656460.0</td>
      <td>CN</td>
      <td>0.00000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5279</td>
      <td>SMC</td>
      <td>68.5</td>
      <td>Male</td>
      <td>20</td>
      <td>Not Hisp/Latino</td>
      <td>White</td>
      <td>Married</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.50629</td>
      <td>0.985156</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>30.0</td>
      <td>61.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>20.0000</td>
      <td>27.0</td>
      <td>1.250</td>
      <td>1.44444</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>1.16667</td>
      <td>1.75</td>
      <td>1.25641</td>
      <td>1.875</td>
      <td>1.22222</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.33333</td>
      <td>1.33333</td>
      <td>1.37838</td>
      <td>0.0</td>
      <td>21327.0</td>
      <td>7654.0</td>
      <td>1081140.0</td>
      <td>4065.0</td>
      <td>17964.0</td>
      <td>18611.0</td>
      <td>1508210.0</td>
      <td>CN</td>
      <td>0.00000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Encoding

We transformed all remaining categorical variables into numerical values. For baseline diagnosis DX_bl, we mapped cognitive normal `CN` to 0, cognitive impairment (`LMCI`, `EMCI`, `SMC`) to 1 and Alzheimer's Disease `AD` to 2. Our task became a multiclass classification problem. For gender `PTGENDER`, we labeled Female as 0, and Male as 1. We used one-hot encoding for all other categorical variables. For `PTETHCAT` and `PTMARRY`, we found there are a few `Unknown`'s inside the data. Since the number of `Unknown`'s is very small, we simply dropped the rows with missing values in these columns to avoid adding another variable. 



```python
adnimerge_clean['DX_bl'] = adnimerge_clean['DX_bl'].map({'CN':0,'AD':2,'EMCI':1,'LMCI':1,'SMC':1})
adnimerge_clean['PTGENDER'] = adnimerge_clean['PTGENDER'].map({'Female':0,'Male':1})
adnimerge_clean = pd.get_dummies(adnimerge_clean, columns=['PTRACCAT'], drop_first=True)
adnimerge_clean = adnimerge_clean[adnimerge_clean.PTETHCAT!='Unknown']
adnimerge_clean = pd.get_dummies(adnimerge_clean, columns=['PTETHCAT'], drop_first=True)
adnimerge_clean = adnimerge_clean[adnimerge_clean.PTMARRY!='Unknown']
adnimerge_clean = pd.get_dummies(adnimerge_clean, columns=['PTMARRY'], drop_first=True)
col_names_replaced = [col.replace(' ', '_') for col in adnimerge_clean.columns]
adnimerge_clean.rename(columns=dict(zip(adnimerge_clean.columns, col_names_replaced)), inplace=True)

print(adnimerge_clean.shape)
adnimerge_clean.head()
```


    (4681, 55)





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
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>CSF_ABETA</th>
      <th>CSF_TAU</th>
      <th>CSF_PTAU</th>
      <th>FDG</th>
      <th>AV45</th>
      <th>CDRSB</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>MMSE</th>
      <th>RAVLT_immediate</th>
      <th>RAVLT_learning</th>
      <th>RAVLT_forgetting</th>
      <th>RAVLT_perc_forgetting</th>
      <th>MOCA</th>
      <th>EcogPtMem</th>
      <th>EcogPtLang</th>
      <th>EcogPtVisspat</th>
      <th>EcogPtPlan</th>
      <th>EcogPtOrgan</th>
      <th>EcogPtDivatt</th>
      <th>EcogPtTotal</th>
      <th>EcogSPMem</th>
      <th>EcogSPLang</th>
      <th>EcogSPVisspat</th>
      <th>EcogSPPlan</th>
      <th>EcogSPOrgan</th>
      <th>EcogSPDivatt</th>
      <th>EcogSPTotal</th>
      <th>FAQ</th>
      <th>Ventricles</th>
      <th>Hippocampus</th>
      <th>WholeBrain</th>
      <th>Entorhinal</th>
      <th>Fusiform</th>
      <th>MidTemp</th>
      <th>ICV</th>
      <th>DX</th>
      <th>Month_bl</th>
      <th>Month</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5282</td>
      <td>1</td>
      <td>66.9</td>
      <td>1</td>
      <td>17</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.13549</td>
      <td>1.326790</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>42.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>20.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.625</td>
      <td>1.33333</td>
      <td>1.14286</td>
      <td>1.2</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.25641</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>7851.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1498720.0</td>
      <td>CN</td>
      <td>0.00000</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>5280</td>
      <td>1</td>
      <td>67.5</td>
      <td>1</td>
      <td>16</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.982566</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>50.0000</td>
      <td>27.0</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.14286</td>
      <td>1.2</td>
      <td>1.66667</td>
      <td>1.50</td>
      <td>1.35897</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CN</td>
      <td>24.32790</td>
      <td>24</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5280</td>
      <td>1</td>
      <td>67.5</td>
      <td>1</td>
      <td>16</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>29.0</td>
      <td>54.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>16.6667</td>
      <td>24.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.2</td>
      <td>1.40000</td>
      <td>1.25</td>
      <td>1.28947</td>
      <td>1.750</td>
      <td>1.11111</td>
      <td>1.33333</td>
      <td>1.2</td>
      <td>NaN</td>
      <td>1.25000</td>
      <td>1.40000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CN</td>
      <td>5.57377</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5280</td>
      <td>1</td>
      <td>67.5</td>
      <td>1</td>
      <td>16</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.25195</td>
      <td>0.983143</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>42.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0000</td>
      <td>28.0</td>
      <td>1.875</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>2.00000</td>
      <td>2.00</td>
      <td>1.43590</td>
      <td>1.625</td>
      <td>1.11111</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.25000</td>
      <td>1.00000</td>
      <td>1.25000</td>
      <td>0.0</td>
      <td>33185.0</td>
      <td>8297.0</td>
      <td>1165500.0</td>
      <td>4946.0</td>
      <td>20147.0</td>
      <td>21194.0</td>
      <td>1656460.0</td>
      <td>CN</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5279</td>
      <td>1</td>
      <td>68.5</td>
      <td>1</td>
      <td>20</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.50629</td>
      <td>0.985156</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>30.0</td>
      <td>61.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>20.0000</td>
      <td>27.0</td>
      <td>1.250</td>
      <td>1.44444</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>1.16667</td>
      <td>1.75</td>
      <td>1.25641</td>
      <td>1.875</td>
      <td>1.22222</td>
      <td>1.28571</td>
      <td>1.0</td>
      <td>1.33333</td>
      <td>1.33333</td>
      <td>1.37838</td>
      <td>0.0</td>
      <td>21327.0</td>
      <td>7654.0</td>
      <td>1081140.0</td>
      <td>4065.0</td>
      <td>17964.0</td>
      <td>18611.0</td>
      <td>1508210.0</td>
      <td>CN</td>
      <td>0.00000</td>
      <td>0</td>
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
    </tr>
  </tbody>
</table>
</div>



## Trend Summary

To deal with the longitudinal component of the data, we decided to add an additional variable for each continuous predictor. These variables are the slopes of observations for all visits of each patient. The slopes indicated the progress or changes of each predictor as time passed by. For example, the negative slope of `Hippocampus` for `AD` patients indicates that the volumne of `Hippocampus` is decreasing over time. If there is only one data point or even no data among all visits of a patient for certain predictor, the slope of this predictor will be `NaN` for this patient.

In this way, we only need to keep the baseline observations and the slopes for every patient, and we still have information about the following visits. We modeled longitudinality.



```python
adnimerge_clean = adnimerge_clean.sort_values(by=['RID', 'Month_bl'])

adnimerge_unique = adnimerge_clean[adnimerge_clean['Month']==0].copy()
cols_for_unique = [
    col for col in adnimerge_clean.columns 
    if (not col.endswith('bl')) \
        and (not col.startswith('DX')) \
        and (not col=='Month') \
        or col=='DX_bl']
adnimerge_unique = adnimerge_unique[cols_for_unique]
adnimerge_unique = adnimerge_unique.rename(index=lambda i:adnimerge_clean['RID'][i])

print(adnimerge_unique.shape)
adnimerge_unique.head()
```


    (783, 52)





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
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>CSF_ABETA</th>
      <th>CSF_TAU</th>
      <th>CSF_PTAU</th>
      <th>FDG</th>
      <th>AV45</th>
      <th>CDRSB</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>MMSE</th>
      <th>RAVLT_immediate</th>
      <th>RAVLT_learning</th>
      <th>RAVLT_forgetting</th>
      <th>RAVLT_perc_forgetting</th>
      <th>MOCA</th>
      <th>EcogPtMem</th>
      <th>EcogPtLang</th>
      <th>EcogPtVisspat</th>
      <th>EcogPtPlan</th>
      <th>EcogPtOrgan</th>
      <th>EcogPtDivatt</th>
      <th>EcogPtTotal</th>
      <th>EcogSPMem</th>
      <th>EcogSPLang</th>
      <th>EcogSPVisspat</th>
      <th>EcogSPPlan</th>
      <th>EcogSPOrgan</th>
      <th>EcogSPDivatt</th>
      <th>EcogSPTotal</th>
      <th>FAQ</th>
      <th>Ventricles</th>
      <th>Hippocampus</th>
      <th>WholeBrain</th>
      <th>Entorhinal</th>
      <th>Fusiform</th>
      <th>MidTemp</th>
      <th>ICV</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4001</th>
      <td>4001</td>
      <td>2</td>
      <td>88.5</td>
      <td>0</td>
      <td>9</td>
      <td>0.0</td>
      <td>105.7</td>
      <td>141.6</td>
      <td>36.2</td>
      <td>1.11537</td>
      <td>1.507200</td>
      <td>5.5</td>
      <td>24.0</td>
      <td>38.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.500</td>
      <td>1.11111</td>
      <td>1.00000</td>
      <td>1.0</td>
      <td>1.00000</td>
      <td>1.00</td>
      <td>1.14286</td>
      <td>3.875</td>
      <td>1.00000</td>
      <td>2.25000</td>
      <td>3.6</td>
      <td>3.66667</td>
      <td>4.00</td>
      <td>2.91667</td>
      <td>22.0</td>
      <td>33609.0</td>
      <td>5532.0</td>
      <td>864483.0</td>
      <td>2995.0</td>
      <td>14530.0</td>
      <td>14249.0</td>
      <td>1255450.0</td>
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
    </tr>
    <tr>
      <th>4004</th>
      <td>4004</td>
      <td>1</td>
      <td>66.8</td>
      <td>0</td>
      <td>14</td>
      <td>0.0</td>
      <td>208.1</td>
      <td>83.3</td>
      <td>33.9</td>
      <td>1.26220</td>
      <td>0.973711</td>
      <td>1.5</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>53.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>69.2308</td>
      <td>27.0</td>
      <td>1.750</td>
      <td>1.66667</td>
      <td>1.85714</td>
      <td>1.4</td>
      <td>1.50000</td>
      <td>2.00</td>
      <td>1.69231</td>
      <td>1.625</td>
      <td>1.22222</td>
      <td>1.28571</td>
      <td>1.8</td>
      <td>1.33333</td>
      <td>2.25</td>
      <td>1.51282</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>4005</th>
      <td>4005</td>
      <td>1</td>
      <td>70.5</td>
      <td>1</td>
      <td>16</td>
      <td>1.0</td>
      <td>139.4</td>
      <td>128.0</td>
      <td>73.9</td>
      <td>1.25009</td>
      <td>1.395770</td>
      <td>3.5</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>29.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>100.0000</td>
      <td>28.0</td>
      <td>2.125</td>
      <td>1.00000</td>
      <td>1.14286</td>
      <td>1.4</td>
      <td>1.33333</td>
      <td>1.50</td>
      <td>1.41026</td>
      <td>3.000</td>
      <td>2.33333</td>
      <td>2.57143</td>
      <td>2.2</td>
      <td>2.66667</td>
      <td>2.25</td>
      <td>2.53846</td>
      <td>8.0</td>
      <td>38294.0</td>
      <td>7207.0</td>
      <td>1181170.0</td>
      <td>4405.0</td>
      <td>22968.0</td>
      <td>22654.0</td>
      <td>1768220.0</td>
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
    </tr>
    <tr>
      <th>4007</th>
      <td>4007</td>
      <td>1</td>
      <td>78.1</td>
      <td>1</td>
      <td>20</td>
      <td>1.0</td>
      <td>144.3</td>
      <td>86.2</td>
      <td>40.5</td>
      <td>1.33645</td>
      <td>1.653660</td>
      <td>0.5</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>36.3636</td>
      <td>24.0</td>
      <td>1.500</td>
      <td>1.00000</td>
      <td>1.14286</td>
      <td>1.0</td>
      <td>1.00000</td>
      <td>1.00</td>
      <td>1.12821</td>
      <td>2.625</td>
      <td>1.55556</td>
      <td>1.83333</td>
      <td>2.6</td>
      <td>1.83333</td>
      <td>2.75</td>
      <td>2.13158</td>
      <td>1.0</td>
      <td>36679.0</td>
      <td>7495.0</td>
      <td>1029740.0</td>
      <td>3522.0</td>
      <td>19848.0</td>
      <td>19938.0</td>
      <td>1426170.0</td>
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
    </tr>
    <tr>
      <th>4009</th>
      <td>4009</td>
      <td>2</td>
      <td>90.3</td>
      <td>1</td>
      <td>17</td>
      <td>0.0</td>
      <td>233.9</td>
      <td>71.4</td>
      <td>22.1</td>
      <td>1.17124</td>
      <td>0.909650</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>25.0</td>
      <td>24.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>60.0000</td>
      <td>18.0</td>
      <td>2.000</td>
      <td>1.33333</td>
      <td>2.14286</td>
      <td>1.8</td>
      <td>2.00000</td>
      <td>1.75</td>
      <td>1.82051</td>
      <td>3.625</td>
      <td>2.44444</td>
      <td>3.71429</td>
      <td>3.4</td>
      <td>3.00000</td>
      <td>2.75</td>
      <td>3.15385</td>
      <td>26.0</td>
      <td>29136.0</td>
      <td>5224.0</td>
      <td>910905.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1338420.0</td>
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
    </tr>
  </tbody>
</table>
</div>





```python
cols_longit = [
    'APOE4', 'CSF_ABETA', 'CSF_TAU', 'CSF_PTAU', 
    'FDG', 'AV45', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE',
    'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
    'RAVLT_perc_forgetting', 'MOCA', 'EcogPtMem', 'EcogPtLang',
    'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
    'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
    'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 'FAQ', 'Ventricles',
    'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']

for c in cols_longit:
    adnimerge_unique[c+'_slope'] = np.full(adnimerge_unique.shape[0], np.nan)
    
for rid in adnimerge_unique.index:
    rows_clean = adnimerge_clean[adnimerge_clean['RID']==rid]
    if rows_clean.shape[0] <= 1:
        continue
    for c in cols_longit:
        if np.isnan(adnimerge_unique.loc[rid, c]):
            continue
        elif np.prod(np.isnan(rows_clean[c].values[1:])) == 1:
            continue
        else:
            idx_available = ~rows_clean[c].isnull()
            xx = rows_clean['Month_bl'][idx_available]
            yy = rows_clean[c][idx_available]
            slope = (np.mean(xx*yy)-np.mean(xx)*np.mean(yy)) / np.var(xx)
            adnimerge_unique.loc[rid, c+'_slope'] = slope
```


## Rearrange and Export

After initial date cleaning, we have 783 observations, one for each patient. To improve readablity, we ordered the columns in a way that the slope of a predictor followed the predictor itself. We output the data to a csv file for further analysis and modeling.



```python
cols_pt = [c for c in adnimerge_unique.columns if c.startswith('PT')]
cols_slope = [c for c in adnimerge_unique.columns if c.endswith('slope')]
cols_contwithslope = [] 
for pair in zip(cols_longit, cols_slope):
    cols_contwithslope += pair

cols_reordered = ['RID','DX_bl','AGE'] + cols_pt + cols_contwithslope
adnimerge_unique = adnimerge_unique[cols_reordered]
adnimerge_unique = adnimerge_unique.rename(columns={'AGE':'PTAGE'})

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
      <th>4001</th>
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
      <th>4004</th>
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
      <th>4005</th>
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
      <th>4007</th>
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
      <th>4009</th>
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
adnimerge_unique.to_csv('data/ADNIMERGE_unique.csv', index=False)
```

