# =====================imports========================================

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Src.RandomForest import model_randomforest;
from Src.Preprocess import preprocess_data_pca;

# ===================================================================


# ======================main program=================================

plt.interactive(False)

# load the provided data
train_features_path = 'Files/dengue_features_train.csv'

train_labels_path = 'Files/dengue_labels_train.csv'

test_data_path = 'Files/dengue_features_test.csv'

submission_format_path = 'Files/submission_format.csv'

# save predictions
test_predictions_path = 'Results/predictions_RandomForest.csv'

sj_train, iq_train = preprocess_data_pca(train_features_path, labels_path=train_labels_path)

sj_best_model = model_randomforest(sj_train)
iq_best_model = model_randomforest(iq_train)

# find predictions
sj_test, iq_test = preprocess_data_pca(test_data_path)

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv(submission_format_path, index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv(test_predictions_path)