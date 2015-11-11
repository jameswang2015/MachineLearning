Besides all the required files, "data_discovery.py"  and "gridSearchTest.py" is added
into the final project.

* data_discovery.py:
it runs the code to discover the given Enron dataset and answers basic questions like:
- how many targets/features are in the dataset
- what features are included and what does a typical record look like
- how many POIs and non_POIs
- features with missing value ("NaN")
- feature selection score by SelectKBest()

the last section is to code and plot the performance score function with the different
number of best features. This is critical in the final algorithm to determine how many
features should be used to get the best performance score.

* gridSearchTest.py:
GridSearchCV is particularly tested in this file and svm.SVC is used as the test
algorithm. scoring is set to "f1" when performing GridSearch. The final best_parameters_
found in this test are used as the parameters in final poi_id.py for svm.SVC algorithm.
The performance scores are improved by using these parameters.