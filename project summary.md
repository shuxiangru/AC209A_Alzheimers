---
title: Project Summary
nav_include: 8
---

## Project Trajectory
Originally, we aimed to investigate whether certain testing methods of Alzheimer’s disease is more accurate for certain group of patients (e.g. White male older than 75) in diagnosing AD. If we were able to do so, doctors could be even more efficient in diagnosing patients with different background. However, since we only had data for 783 patients, we found that we did not have sufficient data enabling us to divide patients into groups. The result of the model will be unreliable if the training data size is very small. 

The patient data has a longitudinal component that adds complexity. We considered treating all observations as independent observations in milestone 3, but later realized that the observations from the same patient are likely to be very highly correlated. We experimented with multiple possible ways to deal with longitudinality, and ended up feature engineering a slope variable for each continuous variable.


## Result and Conclusion

Random forest classifier and linear discriminant analysis have the highest test accuracy, and logistic regression with l1 regularization ranked the third. Random forest classifier is good at dealing with unbalanced and unscaled data, and usually does not have any problem of overfitting. Thus we see high training accuracy and test accuracy using random forest classifer. LDA performs surprisingly well in the classification of Alzheimer's disease. We found that the distribution of each class is very similar for almost all variables, and that may be the reason why LDA acheives high test accuracies in this application.

Our focus is achieving high classification accuracy of Alzheimer's disease. AdaBoost, LDA and decision tree classifier all acheive test accuracy of 88.89% on `AD` patients. Logistic regression with l1 regularization, random forest classifier and multinomial logistic regression reach test accuracy of over 80% on the classification of `AD`. However, at the same time, we do not want to incorrectly classify too many patients from the other two classes. The overall performance suggests that random forest classifier and LDA are the most suitable models in the classification of Alzheimer's disease. 


Significant predictors, minimal number of tests



To further improve the classification accuracy of Alzheimer's disease, we tested Neural Networks model. The overall test accuracy of neural networks is close to that of the random forest. However, we found that the test accuracy for the cognitively normal group and the Alzheimer's disease group is considerably higher using neural network. Hence, we would prefer neural network model to the random forest classifer.


## Future Work

Currently, we are concerned about the accuracy of predicting `AD` patients particularly, so the best model on predicting only `AD` probably does not have a high accuracy on other classes. In the future, we will extend our goal to achieve higher accuracy of predicting all classes by spending more time tuning parameters for existing models or trying new models. 

To further extend this project, we might take one step further and investigate whether certain AD testing method is more accurate for a certain group of patient (e.g. White male older than 75) in diagnosing AD. To do that, we will conduct statistically testings between groups of patients with shared characteristics on the performance of different testing methods. If some patterns are found, doctors can be far more efficient in diagnosing patients from various backgrounds.
