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


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
### compute the ratio of to/from poi against total to/from emails
### this is used for creating new features of ratio_to_poi and ratio_from_poi
def computeRatio(emails, emails_poi):
    ratio = 0.
    if emails != 'NaN' and emails_poi != 'NaN':
        ratio = emails_poi/emails
    return ratio

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return [precision, recall, f1]
    except:
        print "Got a divide by zero when trying out:", clf


### Task 0: load the data
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 1: Remove outliers
# this is an excel spreadsheet quirk that should be removed
data_dict.pop('TOTAL',0)

### print arbitrary record to get an idea of the feature/value
print 'an arbitrary record from the given dataset looks like this:'
print data_dict.itervalues().next(), '\n'
### number of features
all_features = data_dict.itervalues().next().keys()
print 'number of features from Enron dataset is {}.\n'.format(len(all_features))
print 'list of features from given Enron dataset is \n{}.\n'.format(all_features)

### number of records and features
n_total_record = 0
n_POIs = 0
n_Non_POIs = 0
n_NaN_per_feature = {}
for name in data_dict:
    n_total_record += 1
    if data_dict[name]['poi'] == 1:
        n_POIs += 1
    else:
        n_Non_POIs += 1
    for feature in all_features:
        if data_dict[name][feature] == 'NaN':
            if feature not in n_NaN_per_feature:
                n_NaN_per_feature[feature] = 1
            else:
                n_NaN_per_feature[feature] += 1
        
print 'total records from given Enron dataset is {}.\n'.format(n_total_record)
print 'number of POIs is {}.\n'.format(n_POIs)
print 'number of Non_POIs is {}.\n'.format(n_Non_POIs)
print '{} features have missing value (represented as "NaN") and they are listed as below:'\
.format(len(n_NaN_per_feature))
print sorted(n_NaN_per_feature.items(), key = operator.itemgetter(1)), '\n'


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
# features_list = ['poi', 'bonus', 'total_stock_value', 'ratio_to_poi', 'salary', 
#                  'exercised_stock_options']
print 'The Features selected by SelectKBest(k = 3) are:\n{}\n'.format(features_list)
print 'The selection.scores_ is: \n{}\n'.format(selection.scores_)
print 'The feature score list by SelectKBest() is'
all_features.remove('poi')
print sorted(zip(all_features, selection.scores_), key = operator.itemgetter(1)), '\n'
print
### adding 'poi' as the 1st element in features_list before fed into feature_format
features_list.insert(0, 'poi')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# DecisionTree
tree = tree.DecisionTreeClassifier(criterion = 'gini', 
                                   min_samples_split = 2, 
                                   random_state = 10)


y_precision = []
y_recall = []
y_f1 = []
for j in range(21):
    selection = SelectKBest(k = j+1)
    selection.fit(features, labels)
    features_list = [all_features[i] for i in selection.get_support(indices=True)]
    features_list.insert(0, 'poi')
    a, b, c = test_classifier(tree, my_dataset, features_list)
    y_precision.append(a)
    y_recall.append(b)
    y_f1.append(c)

### plot the relation of performance score vs. K
k = np.arange(1,22,1)
plt.plot(k, y_precision, '-ro', label = 'precision')
plt.plot(k, y_recall, '-bs', label = 'recall')
plt.plot(k, y_f1, '-g^', label = 'f1')
plt.xticks(np.arange(1, 22, 1.0))
plt.xlim([0,22])
plt.grid(True)
plt.xlabel('K Best Features by SelectKBest()')
plt.ylabel('Performance Score')
plt.title('Performance Score ~ Number of Features')
plt.legend()
plt.show
plt.savefig('feature.png')