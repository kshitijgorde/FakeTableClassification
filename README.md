# FakeTableClassification
Classification task and plotting F-1 measures from Real tables and Fake/Anonymized tables generated using Generative Adversarial Networks (GAN's)
The GitHub.com/mahmoodm2/datanonym repository contains the following files for classification :

1 - Real Data: Train and Test files with their labels

data/LACity/ LACity_cleaned.pickle = Train Data : A numpy array file containing  train data : 15000 * 23 
data/LACity/test_LACity.cleaned.pickle = Test  Data : A numpy array file containing  test  data : 3750  * 23

data/LACity/LACity_Labels.pickle = Train Label Data : A numpy array file containing  train label data : 15000  0/1 Values
data/LACity/test_LACity_labels.pickle = Test  Data : A numpy array file containing  test label  data : 3750   0/1 Values

2 - Fake Data 1 : Generated from DCGAN model and scaled to original values
Address : /samples/LACity/folder_name 
Each folder such as CO_11_OO contains a scaled_fake_tabular.pickle as a  14976 * 23 matrix.
The name of the folder represents a test id and pleas keep it in your results as a test_id to link the results to their test settings. 
You can generate labels from these files. 
The column I used to generate the labels in real data was column "Total Payment" or column 9 in each  scaled_fake_tabular.pickle
The absolute value to classify the rows is  77636.3656547.  Rows with "Total Payment" value more than this value are labeled as 1 ( Rich) and below this value as 0 ( poor).

3-  Fake Data 2 : Generated from ARX software and cleaned 
Address :  /arx/LACity/Anonymized/arx_testid.pickle. There are 36 files.
 Each files such as ARX_5_N_99.pickle contains  a 15000 * 23 matrix.
The name of the file represents  test settings and pleas keep it in your results as a test_id to link the results to their test settings. 
You can generate labels from these files. 
The column I used to generate labels in real data is column "Total Payment"or column 9 in each *.pickle file.
The absolute value to classify the rows is  77636.3656547. Rows with "Total Payment" value more than this value are labeled as 1( Rich) and below this value as 0 ( poor)


Please check the data and let me know any problems.

The Columns Names are:
selected_columns = [ 'Year', 
       'Projected Annual Salary', 'Q1 Payments',
       'Q2 Payments', 'Q3 Payments', 'Q4 Payments', 'Payments Over Base Pay',
       '% Over Base Pay', 'Total Payments', 'Base Pay', 'Permanent Bonus Pay',
       'Longevity Bonus Pay', 'Temporary Bonus Pay', 'Lump Sum Pay',
       'Overtime Pay', 'Other Pay & Adjustments',
       'Other Pay (Payroll Explorer)',  'FMS Department',
       'Job Class', 'Average Health Cost', 'Average Dental Cost',
       'Average Basic Life', 'Average Benefit Cost', 
       ]