import os
import pandas as pd
import numpy as np
from MyDecisionTreeClassifier import MyClassifiers


dir_name = os.path.dirname(os.path.realpath(__file__))

df = pd.read_pickle('LACityRealData/LACity_cleaned.pickle')
df_labels = pd.read_pickle('LACityRealData/LACity_labels.pickle')

test_df = pd.read_pickle('LACityRealData/test_LACity_cleaned.pickle')
test_df_labels = pd.read_pickle('LACityRealData/test_LACity_labels.pickle')

# Test - 3750
# Train - 15000
clf = MyClassifiers()
print len(df)
print len(df_labels)
print len(test_df)
print len(test_df_labels)
df = pd.np.array(df)
df_labels = pd.np.array(df_labels)
test_df = pd.np.array(test_df)
test_df_labels = pd.np.array(test_df_labels)
clf.classify(df,df_labels,test_df,test_df_labels)



stringTemplate = 'pos1={1} pos2={2} pos3={3} foo={4} bar={5}'
#print formatString(stringTemplate, 'test','a','b','c')