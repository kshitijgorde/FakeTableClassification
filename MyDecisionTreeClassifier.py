from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier


import os
import collections

def formatString(stringTemplate, *args, **kwargs):
    # Replace any positional parameters
    for i in range(0, len(args)):
        tmp = '{%s}' % str(1 + i)
        while True:
            pos = stringTemplate.find(tmp)
            if pos < 0:
                break
            stringTemplate = stringTemplate[:pos] + \
                             str(args[i]) + \
                             stringTemplate[pos + len(tmp):]

    # Replace any named parameters
    for key, val in kwargs.items():
        tmp = '{%s}' % key
        while True:
            pos = stringTemplate.find(tmp)
            if pos < 0:
                break
            stringTemplate = stringTemplate[:pos] + \
                             str(val) + \
                             stringTemplate[pos + len(tmp):]

    return stringTemplate

class MyClassifiers():
    'Handles Predicting Phishing URL by implementing scikit-learn DecisionTree Classifier'
    def classify(self,realTrainFeatures, realTrainLabels, realTest_Features, realTest_Labels):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        #predictionResult = open(dir_name+'/'+technique+'DecisionTreeResultsNoOversampling.txt','a+')
        #predictionResult.truncate()
        algorithms = ['DecisionTree','RandomForest','AdaBoost']
        # Create a dual for loop for everyalgorithm
        for eachAlgorithm in algorithms:
            if eachAlgorithm == 'DecisionTree':
                parameters_decisionTree = {0: "criterion=gini,splitter=best,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,presort=False",
                                           1: "criterion=entropy,splitter=random,max_depth=None,min_samples_split=3,min_samples_leaf=1,max_features=None,presort=False",
                                           2: "criterion=gini,splitter=best,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,presort=False",
                                           3: "criterion=entropy,splitter=random,max_depth=None,min_samples_split=4,min_samples_leaf=1,max_features=None,presort=False",
                                           4: "criterion=gini,splitter=best,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,presort=False"
                                           }
                for i in range(0,5):
                    parameter = parameters_decisionTree.values()[i]
                    param = parameter.split(',')
                    my_criterion = param[0].split('=')[1]
                    my_splitter = param[1].split('=')[1]
                    my_max_depth = param[2].split('=')[1]
                    my_min_samples_split = param[3].split('=')[1]
                    my_min_samples_leaf = param[4].split('=')[1]
                    my_max_features = param[5].split('=')[1]
                    my_presort = param[6].split('=')[1]
                    #Every i represents one parameter setup..Define the parameters
                    clf = DecisionTreeClassifier(criterion=my_criterion,splitter=my_splitter,max_depth=None,min_samples_split=int(my_min_samples_split),
                                                 min_samples_leaf=int(my_min_samples_leaf),presort=my_presort)
                    clf.fit(realTrainFeatures,realTrainLabels)
                    realResult = clf.predict(realTest_Features)
                    f1score = f1_score(realTest_Labels,realResult,average='macro')
                    print "The f1_score is:" + str(f1score)

            if eachAlgorithm == 'AdaBoost':
                parameters_AdaBoost = {
                    0: "n_estimators=50,learning_rate=1,algorithm=SAMME",
                    1: "n_estimators=80,learning_rate=1.4,algorithm=SAMME.R",
                    2: "n_estimators=100,learning_rate=1.3,algorithm=SAMME",
                    3: "n_estimators=120,learning_rate=1.6,algorithm=SAMME.R",
                    4: "n_estimators=250,learning_rate=2.0,algorithm=SAMME"
                    }
                for i in range(0,5):
                    parameter = parameters_AdaBoost.values()[i]
                    param = parameter.split(',')
                    my_n_estimators = int(param[0].split('=')[1])
                    my_learning_rate = float(param[1].split('=')[1])
                    my_algorithm = param[2].split('=')[1]
                    #Every i represents one parameter setup..Define the parameters
                    clf = AdaBoostClassifier(n_estimators=my_n_estimators,learning_rate=my_learning_rate,algorithm=my_algorithm)
                    clf.fit(realTrainFeatures,realTrainLabels)
                    realResult = clf.predict(realTest_Features)
                    f1score = f1_score(realTest_Labels,realResult,average='macro')
                    print "The f1_score is:" + str(f1score)

            if eachAlgorithm == 'RandomForest':
                parameters_RandomForest = {
                    0: "n_estimators=10, criterion=gini, min_samples_split=2,max_features=auto,bootstrap=True,verbose=0, warm_start=False",
                    1: "n_estimators=10, criterion=gini, min_samples_split=2,max_features=auto,bootstrap=True,verbose=0, warm_start=False",
                    2: "n_estimators=10, criterion=gini, min_samples_split=2,max_features=auto,bootstrap=True,verbose=0, warm_start=False",
                    3: "n_estimators=10, criterion=gini, min_samples_split=2,max_features=auto,bootstrap=True,verbose=0, warm_start=False",
                    4: "n_estimators=10, criterion=gini, min_samples_split=2,max_features=auto,bootstrap=True,verbose=0, warm_start=False"
                    }
                for i in range(0,5):
                    parameter = parameters_RandomForest.values()[i]
                    param = parameter.split(',')
                    RF_n_estimators = int(param[0].split('=')[1])
                    RF_criterion = param[1].split('=')[1]
                    RF_min_samples_split = int(param[2].split('=')[1])
                    RF_max_features= param[3].split('=')[1]
                    RF_bootstrap= param[4].split('=')[1]
                    RF_verbose= int(param[5].split('=')[1])
                    RF_warm_start= param[6].split('=')[1]



                    #Every i represents one parameter setup..Define the parameters
                    clf = RandomForestClassifier(n_estimators=RF_n_estimators,criterion=RF_criterion,min_samples_split=RF_min_samples_split,max_features=RF_max_features,bootstrap=RF_bootstrap,verbose=RF_verbose,warm_start=RF_warm_start)
                    clf.fit(realTrainFeatures,realTrainLabels)
                    realResult = clf.predict(realTest_Features)
                    f1score = f1_score(realTest_Labels,realResult,average='macro')
                    print "The f1_score is:" + str(f1score)