---
title: Literature Review
nav_include: 4
---

## Literature Review

To accurately diagnose Alzheimer's Disease (AD) and to determine the most significant tests, we need to first investigate possible predictors for the diagnosis. It's commonly believed that there are four types of factors/test results that are helpful in diagnosing AD: demographic factors, brain imaging results, cognitive test scores, and biomarker measurements. Young et al. [1] demonstrated the changes of biomarker as Alzheimer's disease progresses, which supported our feature engineering decision of adding a slope variable for all continuous predictors. We learned from the paper that the value of cerebrospinal fluid biomarkers, such as amyloid-beta, phosphorylated tau and total tau, would become abnormal if a person had Alzheimer's disease. This motivated us to merge our ADNIMERGE data set with the UPenn CSF biomarkers data set. Young pointed out that the rates of atrophy, cogitive test scores, and regional brain volume would vary greatly between AD patients and cognitively normal people. In addition, Moradi et al. [2] stated that high values of the Rey's Auditory Verbal Learning Test Scores were associated with the Alzheimer's disease. 

Notably, many of the assessments are essentially testing similar aspects of the patients. For example, both the Rey Auditory Verbal Learning Test (RAVLT) and the Alzheimer’s Disease Assessment Scale (ADAS) includes tests for rate of learning and short-term memory [2,3,4]. So it would be helpful if we can reduce the number of tests in the process of disagnosis. Furthermore, if the results of certain tests along several visits are not significantly different or indicative for the diagnosis, we would also like to eliminate them.

In the modeling process, we tried almost all classification models we learned in the class. Even though the acccuracy on the test set is decently high, we would like to further enhance the predicting power. Weiner [5] pointed out in the summary of recent publications that advances in machine learning techniques such as neural networks have improved diagnostic and prognostic accuracy. This inspired us to implement neutal networks.


### Reference
[1] Young, Alexandra L., et al. "A Data-Driven Model Of Biomarker Changes In Sporadic Alzheimer's Disease." Alzheimer's & Dementia, vol. 10, no. 4, 2014, doi:10.1016/j.jalz.2014.04.180.

[2] Moradi, Elaheh et al. "Rey's Auditory Verbal Learning Test Scores Can Be Predicted from Whole Brain MRI in Alzheimer's Disease." NeuroImage : Clinical13 (2017): 415–427. PMC. Web. 26 Nov. 2017.

[3] Rosenberg, Samuel J., Ryan, Joseph J., Prifitera, Aurelio (2009). Rey auditory-verbal learning test performance of patients with and without memory impairment. Journal of Clinical Psychology, 40 (3), 785-787.

[4] Mohs, Richard C., et al. “Development of Cognitive Instruments for Use in Clinical Trials of Antidementia Drugs.” Alzheimer Disease & Associated Disorders, vol. 11, 1997, pp. 13–21., doi:10.1097/00002093-199700112-00003.

[5] Weiner, Michael W. "Recent Publications from the Alzheimer's Disease Neuroimaging Initiative: Reviewing Progress toward Improved AD Clinical Trials." Alzheimer's & Dementia, Elsevier, 22 Mar. 2017.

