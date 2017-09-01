# =====================imports========================================

from __future__ import print_function
from __future__ import division
from sklearn import svm

from warnings import filterwarnings
from sklearn.tree import DecisionTreeClassifier

filterwarnings('ignore')

# ===================================================================


# =================funcion to create model===========================
def model_decision_tree(train):
    Y = train["total_cases"]
    X = train.ix[:, train.columns != "total_cases"]

    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
    fitted_model = model.fit(X, Y)

    return fitted_model
# ===================================================================
