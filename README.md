## Feature-based approach with  Cross-validation for cluster validation 
This notebook contains Feature-based cluster  validation in clustering analysis designed to measure the performance of a clustering model in predicting cluster labels for new data points, given that the model is already constructed from the training data. This approach is based on cross-validation with the following procedures: 
1. Firstly, the new approach calculates the occurrence of features in training and testing samples assigned to the same cluster, in terms of probability score. 
2. Then the distance between the vector of scores in training and testing samples is calculated using some distance measurement metrics (e.g. cosine similarity, mean squared eror, etc.)
