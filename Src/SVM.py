# =====================imports========================================

from __future__ import print_function
from __future__ import division
from sklearn import svm

from warnings import filterwarnings

filterwarnings('ignore')

# ===================================================================


# =================funcion to create model===========================
def model_svm(train):
    Y = train["total_cases"]
    X = train.ix[:, train.columns != "total_cases"]

    model = svm.SVR()
    fitted_model = model.fit(X, Y)

    return fitted_model
# ===================================================================
