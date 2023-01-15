## Label-based approach based on Cross-validation for cluster validation 
The proposed cluster  validation method is used to measure the performance of a clustering model to predict cluster labels for new data points, given that the model is already constructed from the training data.This approach is based on cross-validation with the following procesures: 
 1. Firstly, the new approach calculates the occurrence of features in training and testing samples assigned to the same cluster, in terms of probability score. 
2. Then the distance between the vector of scores in training and testing samples is calculated using the root mean square error (RMSE) or another related distance metric in clusters of every k-fold. 
3. Finally, the global cluster validity index is by summing up all scores across all the clusters to measure the compactness of the defined clusters. 

 The detailed description and implementation procedure for the proposed validation approach is presented as follows: 
 Let:
 F = { λi : i= 1,..., q } :   the set of all features  in a given  dataset
 
 q=|F|:  the number of features in a dataset.
 
 k: the number of folds in the cross-validation procedure,

 C: the number of clusters generated by the clustering algorithm.

 y_m, m = 1,…, q: the probability that a sample from the training dataset assigned to cluster i has the mth  feature 
 
y^_m, m = 1,…, q: the probability that a sample from the testing dataset assigned to cluster i has the mth feature

 1. Shuffle the original dataset randomly 

 2. Split the original dataset into k parts (folds)  # k=10, for 10-fold cross-validation.

3. For each fold j=1,…,k.

 	a) Take fold j as the test dataset (each fold, in turn, is used as the test dataset).
    
	b) Take the remaining folds together as the training dataset.
    
 	c) apply dimensionality reduction (if needed)
    
	d) apply normalization to dataset (if needed) 
    
	e) Generate clusters on the training dataset.
    
	f) Assign data points from the test dataset (selected in step ‘a’) into the corresponding clusters obtained in step ‘e.’
	g) For each cluster i = 1, …, C found in step 'e':
 	   i) Compute the probabilities y_m, m = 1,…, q  of the occurrence of the features  in cluster i based on the samples in the training dataset.       
       ii) Compute the probabilities y ̂_m, m = 1,…, q of the occurrence of the features in cluster i using the assignment of the points from the test dataset to the clusters, which was obtained in step ‘f.’     
  iii) Compute the root mean squared error (MAPEij) between the probabilities calculated in steps ‘a.’ and ‘b.’. Note down the scores/errors as a quality measure for cluster i obtained in fold j.       
 4. When the loop in step 3 finishes (and so every fold served as the test set), take the average over the k folds of the recorded scores for each cluster and/or overall, the clusters (equation (3)). 






