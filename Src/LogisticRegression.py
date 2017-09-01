# =====================imports========================================

from __future__ import print_function
from __future__ import division

from sklearn import linear_model, datasets

from warnings import filterwarnings

filterwarnings('ignore')

# ===================================================================


# =================funcion to create model===========================
def log_reg(X,Y,T):
    h = .02  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)

    Z = logreg.predict(T)

    return Z
# ===================================================================
