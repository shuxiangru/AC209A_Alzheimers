---
title: Project Summary
nav_include: 8
---

## Contents
{:.no_toc}
*  
{: toc}


## Project Trajectory
Originally, we aimed to investigate whether certain testing methods of Alzheimer’s disease is more accurate for certain group of patients (e.g. White male older than 75) in diagnosing AD. If we were able to do so, doctors could be even more efficient in diagnosing patients with different background. However, since we only had data for 783 patients, we found that we did not have sufficient data enabling us to divide patients into groups. The result of the model will be unreliable if the training data size is very small. 

The patient data has a longitudinal component that adds complexity. We considered treating all observations as independent observations in milestone 3, but later realized that the observations from the same patient were likely to be very highly correlated. We experimented with multiple possible ways to deal with longitudinality, and ended up feature engineering a slope variable for each continuous variable.


## Result and Conclusion

Random forest classifier, linear discriminant analysis and logistic regression with l1 regularization have the highest test accuracy. Random forest classifier is good at dealing with unbalanced and unscaled data, and usually does not have any problem of overfitting. Thus we see high training accuracy and test accuracy using random forest classifer. LDA performs surprisingly well in the classification of Alzheimer's disease. We found that the distribution of each class (`AD` Alzheimer's disease, `CI` cognitive impairment and `CN` cognitively normal) is very similar for almost all variables, and that may be the reason why LDA acheives high test accuracies in this application.

Our main focus is to achieve high classification accuracy on the Alzheimer's disease. Random forest classifier has the highest test accuracy on Alzheimer's disease patients. Adaboost, LDA, decision tree, logistic regression with l1 regularization and multinomial logistic regression all achieve test accuracy of over 0.85 on the classification of `AD`. However, at the same time, we do not want to incorrectly classify too many patients from the other two classes. We divided all models into three groups according to test accuracies. The overall performance suggests that Type III models such as random forest classifier and LDA are the most suitable models in the classification of the Alzheimer's disease. 

After performing boostrapping and forward/backward variable selections, we ended up with a set of the most significant predictors. According to those predictors, the tests completed excluded is `MOCA`(Montreal Cognitive Assessment). The result of this test is not significant in the diagnosis, possibly because other assessments are testing similar aspects of the patients.

The factors that are significant themselves but insignificant in their slopes are `MMSE`, `EcogSPMem`, `ADAS13`, `EcogPtMem`, `CSF_TAU`, `FAQ`, `EcogPtLang`, and `EcogSPPlan`. This result indicates that tests associated with these factors only need to be conducted once at the baseline visit and are not necessary in the following visits. Specifically, Mini-Mental State Examination (`MMSE`), Everyday Cognition test on Participant Memory and Language, Study Partner Memory and Plan (`EcogSPMem`, `EcogPtMem`, `EcogPtLang`, `EcogSPPlan`), Functional Activities Questionnaire in Older Adults with Dementia (`FAQ`), Alzheimer’s Disease Assessment Scale (`ADAS13`) and biosample test (`CSF_TAU`) only need to be checked at the first visit.

The factors that are significant in their slopes are `AV45_slope`, `EcogSPLang_slope`, `WholeBrain_slope`, `MidTemp_slope`, `Fusiform_slope`, `FDG_slope`, `RAVLT_perc_forgetting_slope` and `ICV_slope`. That shows that the change in these parameters within two years after the first visit is quite important for the diagnosis. So testing these parameters on each subsequent visit is necessary. Specifically, The tests that need to be conducted in every visit are Ecog (Everyday Cognition) test on study partner language, AV45 test, FDG imaging test, RAVLT_learning test and MRI test.

Other significant demographic factors are age, marital status, race and ethnicities. Remarkably, `MMSE` appears in the output of all variable selection methods. So we can conclude that it's an essetial test in the diagnosis of AD.

### Conclusion Tables
#### Advice on Tests

| Advice                                | Tests                                                                                  |
|--------------------------------------:| --------------------------------------------------------------------------------------:|
| Tests not needed                      | MOCA                                                                                   |
| Tests only needed for the first visit | MMSE, ADAS13, FAQ, CSF(TAU),  EcogPtMem, EcogPtLang,  EcogSPPlan, EcogSPMem            |
| Test needed for each following visit  | AV45, RAVLT(perc_forgetting),  FDG, EcogSPLang MRI(WholeBrain, MidTemp, Fusiform, ICV) |
| Tests selected by all models          | MMSE                                                             


#### Predictor Dictionary

| Predictor name | Test subject details                        |
|----------------|---------------------------------------------|
| CSF            | Cerebrospinal fluid                         |
| FDG            | Fluorodeoxyglucose PET                      |
| AV45           | Florbetapir  PET                            |
| ADAS13         | Alzheimer’s Disease Assessment Scale        |
| MMSE           | Mini-Mental State Examination               |
| RAVLT          | Rey Auditory Verbal Learning Immediate Test |
| MOCA           | Montreal Cognitive Assessment               |
| EcogPt         | Everyday Cognition Test on participant      |
| EcogSP         | Everyday Cognition Test on study partner    |
| FAQ            | Functional Activities Questionnaire         |
| MRI            | Magnetic Resonance Imaging                  |

To further improve the classification accuracy of Alzheimer's disease, we tested the neural networks model. The overall test accuracy of neural networks is better than that of the random forest classifier. It also has a significantly higher accuracy on cognitively normal people although its accuracies on cognitive impairment and Alzheimer's disease are slightly lower. We would say that the neural networks model belongs to the group of Type III classifiers. It is also a very promising model.


## Future Work

Currently, we are mostly concerned about the accuracy of classifying Alzheimer's disease, so the best model on predicting Alzheimer's probably does not have a high accuracy on other classes. In the future, we will extend our goal to achieve higher accuracy of predicting all classes by spending more time tuning parameters for existing models and trying new models. 

To further extend this project, we might take one step further and investigate our original goal, i.e. whether certain AD testing method is more accurate for a certain group of patient (e.g. White male older than 75) in diagnosing AD. To do that, we need to collect more medical data for different kinds of patients. We will conduct statistical testings on the performance of different testing methods between groups of patients with shared characteristics. If some patterns are found, doctors can be far more efficient in diagnosing patients from various backgrounds.
