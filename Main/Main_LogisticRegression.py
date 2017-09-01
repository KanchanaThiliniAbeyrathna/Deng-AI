# =====================imports========================================

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

from warnings import filterwarnings

from matplotlib import pyplot as plt

from Src.LogisticRegression import log_reg;
from Src.Preprocess import preprocess_data;

filterwarnings('ignore')

# ===================================================================



# ======================main program=================================

# select features we want
features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k']

plt.interactive(False)

# load the provided data
train_features_path = 'Files/dengue_features_train.csv'

train_labels_path = 'Files/dengue_labels_train.csv'

test_data_path = 'Files/dengue_features_test.csv'

submission_format_path = 'Files/submission_format.csv'

# save predictions
test_predictions_path = 'Results/predictions_LogisticRegression.csv'


sj_train, iq_train = preprocess_data(features,train_features_path, labels_path=train_labels_path)

Y_sj = sj_train["total_cases"]
X_sj = sj_train[features]

Y_iq = iq_train["total_cases"]
X_iq = iq_train[features]

# find predictions
sj_test, iq_test = preprocess_data(features,test_data_path)

sj_predictions = log_reg(X_sj,Y_sj,sj_test).astype(int)
iq_predictions = log_reg(X_iq,Y_iq,iq_test).astype(int)

submission = pd.read_csv(submission_format_path, index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv(test_predictions_path)
