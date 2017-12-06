---
title: Literature Review
nav_include: 2
---

## Literature Review

To accurately classify the Alzheimer's disease, we need to understand what the possible predictors could be. Young et al. [1] demonstrated the changes of biomarker as Alzheimer's disease progresses, which supported our feature engineering decision of adding a slope variable for all continuous predictors. We learned from the paper that the value of cerebrospinal fluid biomarkers, such as amyloid-beta, phosphorylated tau and total tau, would become abnormal if a person had Alzheimer's disease. This motivated us to merge our ADNIMERGE data set with the UPenn CSF biomarkers data set. Young pointed out that the rates of atrophy, cogitive test scores, and regional brain volume will vary greatly between AD patients and cognitively normal people. In addition, Moradi et al. [2] stated that high values of the Rey's Auditory Verbal Learning Test Scores are associated with the Alzheimer's disease.

In the modeling process, we tried almost all classification models we learned in the class. Even though the acccuracy on the test set is decently high, we would like to further enhance the predicting power. Weiner [3] pointed out in the summary of recent publications that advances in machine learning techniques such as neural networks have improved diagnostic and prognostic accuracy. This inspired our implementation of neutal networks.


### Reference
[1]. Young, Alexandra L., et al. "A Data-Driven Model Of Biomarker Changes In Sporadic Alzheimer's Disease." Alzheimer's & Dementia, vol. 10, no. 4, 2014, doi:10.1016/j.jalz.2014.04.180.
[2]. Moradi, Elaheh et al. "Rey's Auditory Verbal Learning Test Scores Can Be Predicted from Whole Brain MRI in Alzheimer's Disease." NeuroImage : Clinical13 (2017): 415–427. PMC. Web. 26 Nov. 2017.
[3]. Weiner, Michael W. "Recent Publications from the Alzheimer's Disease Neuroimaging Initiative: Reviewing Progress toward Improved AD Clinical Trials." Alzheimer's & Dementia, Elsevier, 22 Mar. 2017.
