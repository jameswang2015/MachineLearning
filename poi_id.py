#!/usr/bin/python

from __future__ import division
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### compute the ratio of to/from poi against total to/from emails
### this is used for creating new features of ratio_to_poi and ratio_from_poi
def computeRatio(emails, emails_poi):
    ratio = 0.
    if emails != 'NaN' and emails_poi != 'NaN':
        ratio = emails_poi/emails
    return ratio

### Task 0: load the data
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 1: Remove outliers
# this is an excel spreadsheet quirk that should be removed
data_dict.pop('TOTAL',0)

### Task 2: Create new feature(s)
'''
create new feature of ratio_to_poi which is the ratio of the emails sent from 
this person to any poi against the total emails sent from this person.
Similarly, create new feature of ratio_from_poi which is the ratio of the emails
from any poi to this person against the total emails received by this person.
'''
for key in data_dict:
    ### calculate ratio_to_poi and then add the new feature to data_dict
    from_messages = data_dict[key]['from_messages']    
    from_this_person_to_poi = data_dict[key]['from_this_person_to_poi']
    ratio_to_poi = computeRatio(from_messages, from_this_person_to_poi)
    data_dict[key]['ratio_to_poi'] = ratio_to_poi
    
    ### calculate ratio_from_poi and then add the new feature to data_dict
    to_messages = data_dict[key]['to_messages']
    from_poi_to_this_person = data_dict[key]['from_poi_to_this_person']
    ratio_from_poi = computeRatio(to_messages, from_poi_to_this_person)
    data_dict[key]['ratio_from_poi'] = ratio_from_poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 3: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### I use SelectKBest(k = 5) to select the top 5 features.
### Before applying SelectKBest, we need to 
### extract all features from dataset and save it to all_features
all_features = data_dict.itervalues().next().keys()
# 'email_address' need to be removed before fed into featureFormat()
all_features.remove('email_address') 
# make sure 'poi' is the first feature in the list before fed into feature_format
all_features.remove('poi')
all_features.insert(0, 'poi')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
selection = SelectKBest(k = 3)
selection.fit(features, labels)
# get_support indices need to add 1 to skip 'poi' which is the 1st element in all_features.
features_list = [all_features[i] for i in selection.get_support(indices=True)+1]
# here is the features_list selected by applying SelectKBest():
# features_list = ['poi', 'bonus', 'total_stock_value', 'exercised_stock_options']
print 'The Features selected by SelectKBest(k = 3) are: \n', features_list
print 'The selection.scores_ is: \n', selection.scores_
### adding 'poi' as the 1st element in features_list before fed into feature_format
features_list.insert(0, 'poi')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### GaussianNB
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

# PCA + NB
pca_NB = PCA(n_components = 1)
nb = GaussianNB()
clf_pipe_NB = Pipeline([('pca', pca_NB),
                        ('nb', nb)])

### decision tree algorithm
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(criterion = 'entropy', 
                                       min_samples_split = 2, 
                                       random_state = 10)

# PCA + DecisionTree
'''
NOTE: this algorithm shows the best performance in items of recall and F1 value.
Hence, this is my final algorithm used for this project. By default, this is the
altorighm that's input into the final tester.py function.
'''
pca_tree = PCA(n_components = 2)
tree_1 = tree.DecisionTreeClassifier(criterion = 'gini', 
                                     min_samples_split = 2, 
                                     random_state = 10)
clf_pipe_tree = Pipeline([('pca', pca_tree),
                          ('tree', tree_1)])
                          
# Standardization + PCA + DecisionTree
# feature is normalized prior to PCA processing
# StandardScaler is used as feature normalization
scaler_std = StandardScaler()
pca_tree = PCA(n_components = 2)
tree_2 = tree.DecisionTreeClassifier(criterion = 'gini', 
                                     min_samples_split = 2, 
                                     random_state = 10)
clf_pipe_tree_std = Pipeline([('scaler_std', scaler_std),
                              ('pca', pca_tree),
                              ('tree', tree_2)])

# GridSearchCV used for PCA + DecisionTree
pca_gs = PCA()
tree_3 = tree.DecisionTreeClassifier(random_state = 10)
pipe_tree_gs = Pipeline([('pca_gs', pca_gs),
                         ('tree', tree_3)])
parameters = {'pca_gs__n_components':[1, 2, 3],
              'tree__criterion': ['gini', 'entropy'],
              'tree__min_samples_split': [2, 3, 4]}
clf_pipe_tree_gs = GridSearchCV(pipe_tree_gs,
                                parameters,
                                scoring = 'recall')


### SVM algorithm
from sklearn import svm
# MinMaxScaler is used for feature scaling
scaler_svm = MinMaxScaler()
svc = svm.SVC(kernel = 'sigmoid',C = 16, gamma = 0.2)
clf_pipe_SVM = Pipeline([('scaler_svc', scaler_svm),
                         ('svc', svc)])

              
### K nearest neighbors
# MinMaxScaler is used for feature scaling
from sklearn import neighbors
scaler_knn = MinMaxScaler()
knn = neighbors.KNeighborsClassifier(3, weights = 'uniform')
clf_pipe_knn = Pipeline([('scaler_knn', scaler_knn),
                         ('knn', knn)])


### RandomForest
from sklearn.ensemble import RandomForestClassifier
clf_RandomForest = RandomForestClassifier(n_estimators = 17,
                                          criterion = 'gini', 
                                          min_samples_split = 2, 
                                          random_state = 10)

# PCA + RF
pca_RF = PCA(n_components = 2)
random_forest = RandomForestClassifier(n_estimators = 17,
                                       criterion = 'gini', 
                                       min_samples_split = 2, 
                                       random_state = 10)
clf_pipe_RandomForest = Pipeline([('pca', pca_RF),
                                  ('rf', random_forest)])


### Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf_AdaBoost = AdaBoostClassifier(n_estimators = 25,
                                  algorithm = 'SAMME.R',
                                  random_state = 10)

# PCA + Adaboost
pca_Adaboost = PCA(n_components = 2) 
adaboost = AdaBoostClassifier(n_estimators = 3,
                              algorithm = 'SAMME.R',
                              random_state = 10)
clf_pipe_Adaboost = Pipeline([('pca', pca_Adaboost),
                              ('adaboost', adaboost)])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf_pipe_tree, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf_pipe_tree, my_dataset, features_list)