import os
import pandas as pd
import numpy as np


dir_name = os.path.dirname(os.path.realpath(__file__))

df = pd.read_pickle('LACityRealData/LACity_cleaned.pickle')
df_labels = pd.read_pickle('LACityRealData/LACity_labels.pickle')

test_df = pd.read_pickle('LACityRealData/test_LACity_cleaned.pickle')
test_df_labels = pd.read_pickle('LACityRealData/test_LACity_labels.pickle')

# Test - 3750
# Train - 15000


