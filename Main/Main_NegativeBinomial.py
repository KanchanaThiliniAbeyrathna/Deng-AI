# =====================imports========================================

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

from warnings import filterwarnings

from matplotlib import pyplot as plt

from Src.NegativeBinomial import negative_binomial_model;
from Src.Preprocess import preprocess_data;

filterwarnings('ignore')

# ===================================================================



# ======================main program=================================

# select features we want
features = ['reanalysis_specific_humidity_g_per_kg',
            'reanalysis_dew_point_temp_k',
            'reanalysis_max_air_temp_k',
            'station_min_temp_c']

plt.interactive(False)

# load the provided data
train_features_path = 'Files/dengue_features_train.csv'

train_labels_path = 'Files/dengue_labels_train.csv'

test_data_path = 'Files/dengue_features_test.csv'

submission_format_path = 'Files/submission_format.csv'

# save predictions
test_predictions_path = 'Results/predictions_NegativeBinomial.csv'

sj_train, iq_train = preprocess_data(features,train_features_path, labels_path=train_labels_path)

sj_train_subtrain = sj_train.head(750)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 750)

iq_train_subtrain = iq_train.head(350)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 350)

sj_best_model = negative_binomial_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = negative_binomial_model(iq_train_subtrain, iq_train_subtest)

sj_test, iq_test = preprocess_data(features,test_data_path)

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv(submission_format_path, index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv(test_predictions_path)
