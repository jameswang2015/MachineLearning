Besides all the required files, "data_discovery.py" is added into the final project.
it runs the code to discover the given Enron dataset and answers basic questions like: 
- how many targets/features are in the dataset
- what features are included and what does a typical record look like
- how many POIs and non_POIs
- features with missing value ("NaN")
- feature selection score by SelectKBest()

the last section is to code and plot the performance score function with the different
number of best features. This is critical in the final algorithm to determine how many
features should be used to get the best performance score.