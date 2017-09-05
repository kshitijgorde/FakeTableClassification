import os
import pandas as pd
import numpy as np
from MyDecisionTreeClassifier import MyClassifiers


dir_name = os.path.dirname(os.path.realpath(__file__))

df = pd.read_pickle('LACityRealData/train_LACity_cleaned.pickle')
df_labels = pd.read_pickle('LACityRealData/train_LACity_labels.pickle')

test_df = pd.read_pickle('LACityRealData/test_LACity_cleaned.pickle')
test_df_labels = pd.read_pickle('LACityRealData/test_LACity_labels.pickle')

print('Before Dropping:')
print(df.columns)


# df.drop('Year',axis=1,inplace=True)
# test_df.drop('Year',axis=1,inplace=True)

df.drop('Permanent Bonus Pay',axis=1,inplace=True)
test_df.drop('Permanent Bonus Pay',axis=1,inplace=True)

df.drop('Other Pay (Payroll Explorer)',axis=1,inplace=True)
test_df.drop('Other Pay (Payroll Explorer)',axis=1,inplace=True)

# df.drop('Average Benefit Cost',axis=1,inplace=True)
# test_df.drop('Average Benefit Cost',axis=1,inplace=True)

# df.drop('Average Basic Life',axis=1,inplace=True)
# test_df.drop('Average Basic Life',axis=1,inplace=True)


# df.drop('Job Class',axis=1,inplace=True)
# test_df.drop('Job Class',axis=1,inplace=True)

# df.drop('FMS Department',axis=1,inplace=True)
# test_df.drop('FMS Department',axis=1,inplace=True)

# df.drop('Other Pay & Adjustments',axis=1,inplace=True)
# test_df.drop('Other Pay & Adjustments',axis=1,inplace=True)

# df.drop('Overtime Pay',axis=1,inplace=True)
# test_df.drop('Overtime Pay',axis=1,inplace=True)

# df.drop('Lump Sum Pay',axis=1,inplace=True)
# test_df.drop('Lump Sum Pay',axis=1,inplace=True)

df.drop('Temporary Bonus Pay',axis=1,inplace=True)
test_df.drop('Temporary Bonus Pay',axis=1,inplace=True)

# df.drop('Longevity Bonus Pay',axis=1,inplace=True)
# test_df.drop('Longevity Bonus Pay',axis=1,inplace=True)

df.drop('% Over Base Pay',axis=1,inplace=True)
test_df.drop('% Over Base Pay',axis=1,inplace=True)

df.drop('Q4 Payments',axis=1,inplace=True)
test_df.drop('Q4 Payments',axis=1,inplace=True)

df.drop('Q3 Payments',axis=1,inplace=True)
test_df.drop('Q3 Payments',axis=1,inplace=True)

df.drop('Q2 Payments',axis=1,inplace=True)
test_df.drop('Q2 Payments',axis=1,inplace=True)

df.drop('Q1 Payments',axis=1,inplace=True)
test_df.drop('Q1 Payments',axis=1,inplace=True)

df.drop('Projected Annual Salary',axis=1,inplace=True)
test_df.drop('Projected Annual Salary',axis=1,inplace=True)



#-----------------

df.drop('Total Payments',axis = 1,inplace=True)
#del df[unicode('Total Payments')]
test_df.drop('Total Payments',axis = 1,inplace=True)
#del test_df[unicode('Total Payments')]

df.drop('Payments Over Base Pay',axis = 1,inplace=True)
#del df[unicode('Payments Over Base Pay')]

test_df.drop('Payments Over Base Pay',axis = 1,inplace=True)
#del test_df[unicode('Payments Over Base Pay')]

df.drop('Base Pay',axis = 1,inplace=True)
test_df.drop('Base Pay',axis = 1,inplace=True)
#del df[unicode('Base Pay')]
#del test_df[unicode('Base Pay')]
#test_df.drop(unicode('Total Payments'))
print('After dropping')
print(len(df.columns))




# Test - 3750
# Train - 15000
clf = MyClassifiers()
print(len(df))
print(len(df_labels))
print(len(test_df))
print(len(test_df_labels))
df = pd.np.array(df)
df_labels = pd.np.array(df_labels)
test_df = pd.np.array(test_df)
test_df_labels = pd.np.array(test_df_labels)
clf.classify(df,df_labels,test_df,test_df_labels)
