---
title: Project Summary
nav_include: 8
---

## Project Trajectory
Originally, we aimed to investigate whether certain testing methods of Alzheimer’s disease is more accurate for certain group of patients (e.g. White male older than 75) in diagnosing AD. If we were able to do so, doctors could be even more efficient in diagnosing patients with different background. However, since we only had data for 783 patients, we found that we did not have sufficient data enabling us to divide patients into groups. The result of the model will be unreliable if the training data size is very small. 

The patient data has a longitudinal component that adds complexity. We considered treating all observations as independent observations in milestone 3, but later realized that the observations from the same patient are likely to be very highly correlated. We experimented with multiple possible ways to deal with longitudinality, and ended up feature engineering a slope variable for each continuous variable.


## Result and Conclusion

Random forest classifier, linear discriminant analysis and logistic regression with l1 regularization have the highest test accuracy. Random forest classifier is good at dealing with unbalanced and unscaled data, and usually does not have any problem of overfitting. Thus we see high training accuracy and test accuracy using random forest classifer. LDA performs surprisingly well in the classification of Alzheimer's disease. We found that the distribution of each class is very similar for almost all variables, and that may be the reason why LDA acheives high test accuracies in this application.

Our focus is achieving high classification accuracy of Alzheimer's disease. Random forest classifier has the highest test accuracy on Alzheimer's disease patients. Adaboost, LDA, decision tree, and multinomial logistic regression all achieve test accuracy of 0.90 on the classification of AD. However, at the same time, we do not want to incorrectly classify too many patients from the other two classes. We divided all models into three groups according to test accuracies. The overall performance suggests that Type III models such as random forest classifier and LDA are the most suitable models in the classification of Alzheimer's disease. 

After performing boostraping and forward/backward variable selection, we ended up with a set of the most significant predictors. According to those predictors, the tests completed excluded are: `FDG`(fluorodeoxyglucose) PET imaging test, and `MOCA`(Montreal Cognitive Assessment), which mean the results of these tests are not significant in the diagnosis, possibly because other assessments are testing similar aspects of the patients.

The factors that are significant themselves but insignificant in their slopes are: `MMSE`, `EcogSPMem`, `ADAS13`, `EcogPtMem`, `CSF_TAU`, `FAQ`, `EcogPtLang`, and `EcogSPPlan`, which indicate that tests associated with these factors only need to be conducted once at baseline visit and are not necessary in the following visits. Specifically, Mini-Mental State Examination(`MMSE`), Everyday Cognition test on Participant Memory and Language, Study Partner Memory and Plan(`EcogSPMem`, `EcogPtMem`, `EcogPtLang`, `EcogSPPlan`), Functional Activities Questionnaire in Older Adults with Dementia(`FAQ`), Alzheimer’s Disease Assessment Scale(`ADAS13`) and biosample test(`CSF_TAU`) only need to be checked at the first visit.

The factors that are significant in their slopes are: `AV45_slope`, `EcogSPLang_slope`, `EcogPtLang_slope`, `WholeBrain_slope`, `MidTemp_slope`, `Fusiform_slope`, `FDG_slope`, `RAVLT_perc_forgetting_slope` and `ICV_slope`, which mean the change in these parameters within two years after the first visit is quite important for the diagnosis. So testing these parameters on each subsequent visit is necessary. Specifically, The tests that need to be conducted in every visit are Ecog (Everyday Cognition)test,AV45 test, FDG imaging test, RAVLT_learning test and MRI test.

Other significant demographic factors are: `PTAGE`(age), `PTMARRY_Married`, `PTMARRY_Never_married`, `PTETHCAT_Not_Hisp/Latino`, `PTRACCAT_Unknown`, and `PTRACCAT_Hawaiian/Other_PI`. 

Remarkably, `MMSE` appears in the all the variable selection procedures above. So we can conclude that it's a very essetial test in diagnosis of AD.

To further improve the classification accuracy of Alzheimer's disease, we tested Neural Networks model. The overall test accuracy of neural networks is better than that of random forest classifier. It also has a significantly higher accuracy on cognitively normal people although its accuracy on cognitive impairment and Alzheimer's disease is slightly lower. We would say that neural networks belong to the group of Type III classifiers. It is also a very promising model.


## Future Work

Currently, we are concerned about the accuracy of predicting `AD` patients particularly, so the best model on predicting only `AD` probably does not have a high accuracy on other classes. In the future, we will extend our goal to achieve higher accuracy of predicting all classes by spending more time tuning parameters for existing models or trying new models. 

To further extend this project, we might take one step further and investigate whether certain AD testing method is more accurate for a certain group of patient (e.g. White male older than 75) in diagnosing AD. To do that, we will conduct statistically testings between groups of patients with shared characteristics on the performance of different testing methods. If some patterns are found, doctors can be far more efficient in diagnosing patients from various backgrounds.
