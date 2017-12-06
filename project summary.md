---
title: Project Summary
nav_include: 8
---

## Project Trajectory
Originally, we aimed to investigate whether certain testing methods of Alzheimer’s disease is more accurate for certain group of patients (e.g. White male older than 75) in diagnosing AD. If we were able to do so, doctors could be even more efficient in diagnosing patients with different background. However, since we only had data for 783 patients, we found that we did not have sufficient data enabling us to divide patients into groups. The result of the model will be unreliable if the training data size is very small. 

The patient data has a longitudinal component that adds complexity. We considered treating all observations as independent observations in milestone 3, but later realized that the observations from the same patient are likely to be very highly correlated. We experimented with multiple possible ways to deal with longitudinality, and ended up feature engineering a slope variable for each continuous variable.


## Result and Conclusion
Model comparison, best models

Significant predictors, minimal number of tests

209 methods

## Future Work

Currently, we are concerned about the accuracy of predicting `AD` patients particularly, so the best model on predicting only `AD` probably does not have a high accuracy on other classes. In the future, we will extend our goal to achieve higher accuracy of predicting all classes by spending more time tuning parameters for existing models or trying new models. 

To further extend this project, we might take one step further and investigate whether certain AD testing method is more accurate for a certain group of patient (e.g. White male older than 75) in diagnosing AD. To do that, we will conduct statistically testings between groups of patients with shared characteristics on the performance of different testing methods. If some patterns are found, doctors can be far more efficient in diagnosing patients from various backgrounds.
