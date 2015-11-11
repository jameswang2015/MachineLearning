#!/usr/bin/python

from __future__ import division
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle
import operator
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.grid_search import GridSearchCV
from time import time

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

###test GridSearchCV
from sklearn import svm
clf_params= {
                       'clf__C': [1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
                       'clf__gamma': [0.0],
                       'clf__kernel': ['linear', 'poly', 'rbf'],
                       'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
                       'clf__class_weight': [{True: 12, False: 1},
                                               {True: 10, False: 1},
                                               {True: 8, False: 1},
                                               {True: 15, False: 1},
                                               {True: 4, False: 1},
                                               'auto', None]
                      }
                      
#For this Pipeline:
pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()), ('clf', svm.SVC())])
cv = StratifiedShuffleSplit(labels,n_iter = 50,random_state = 42)
a_grid_search = GridSearchCV(pipe, param_grid = clf_params,cv = cv, scoring = 'f1')
t1 = time()
a_grid_search.fit(features,labels)
print 'time for this GridSearchCV is', time() - t1
# pick a winner
best_clf = a_grid_search.best_estimator_
best_param = a_grid_search.best_params_
best_score = a_grid_search.best_score_
print '\nBest estimator is ', best_clf
print '\nBest parameter set is', best_param
print '\nBest score is ', best_score
'''
scoring = 'f1'
C:\Users\James\Anaconda\lib\site-packages\sklearn\metrics\classification.py:958:
 UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no p
redicted samples.
  'precision', 'predicted', average, warn_for)
time for this GridSearchCV is 2059.11099982

Best estimator is  Pipeline(steps=[('minmaxer', MinMaxScaler(copy=True, feature_
range=(0, 1))), ('clf', SVC(C=1, cache_size=200, class_weight={False: 1, True: 8
}, coef0=0.0,
  degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.01, verbose=False))])

Best parameter set is {'clf__gamma': 0.0, 'clf__tol': 0.01, 'clf__C': 1, 'clf__c
lass_weight': {False: 1, True: 8}, 'clf__kernel': 'rbf'}

Best score is  0.40586002886
'''

'''
scoring = 'recall'
Best estimator is  Pipeline(steps=[('minmaxer', MinMaxScaler(copy=True, feature_
range=(0, 1))), ('clf', SVC(C=1e-05, cache_size=200, class_weight={False: 1, Tru
e: 12}, coef0=0.0,
  degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.1, verbose=False))])

Best parameter set is {'clf__gamma': 0.0, 'clf__tol': 0.1, 'clf__C': 1e-05, 'clf
__class_weight': {False: 1, True: 12}, 'clf__kernel': 'linear'}

Best score is  1.0
'''