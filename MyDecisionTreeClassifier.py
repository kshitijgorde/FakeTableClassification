from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np


import os
import collections
class MyClassifiers():
    'Handles Predicting Phishing URL by implementing scikit-learn DecisionTree Classifier'
    def classify(self,realTrainFeatures, realTrainLabels, realTest_Featues, realTest_Labels):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        #predictionResult = open(dir_name+'/'+technique+'DecisionTreeResultsNoOversampling.txt','a+')
        #predictionResult.truncate()
        algorithms = ['DecisionTree','RandomForest','AdaBoost']
        # Create a dual for loop for everyalgorithm
        for eachAlgorithm in algorithms:
            if eachAlgorithm == 'DecisionTree':
                parameters_decisionTree = {0: "criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,presort=False",
                                           1: "criterion='entropy',splitter='random',max_depth=None,min_samples_split=3,min_samples_leaf=1,max_features=None,presort=False",
                                           2: "criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,presort=False",
                                           3: "criterion='entropy',splitter='random',max_depth=None,min_samples_split=4,min_samples_leaf=1,max_features=None,presort=False",
                                           4: "criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,presort=False",
                                           0: "criterion='entropy',splitter='random',max_depth=None,min_samples_split=4,min_samples_leaf=1,max_features=None,presort=False"}
                for i in range(0,4):
                    #Every i represents one parameter setup..Define the parameters











        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20)
            estimator = DecisionTreeClassifier()
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(URL_Train,Label_Train)
            result = clf.predict(URL_Test)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification without Oversampling Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))