import numpy as np
from scipy.spatial.distance import pdist
from src.modelspecific.iscolinear import iscolinear

def homography_degen(X):
    # Extract x1 and x2 from X
    x1 = X[:3, :]  # 3xN
    x2 = X[3:6, :]  # 3xN

    # Check for collinearity for each combination
    r = (iscolinear(x1[:, 0], x1[:, 1], x1[:, 2]) or
         iscolinear(x1[:, 0], x1[:, 1], x1[:, 3]) or
         iscolinear(x1[:, 0], x1[:, 2], x1[:, 3]) or
         iscolinear(x1[:, 1], x1[:, 2], x1[:, 3]) or
         iscolinear(x2[:, 0], x2[:, 1], x2[:, 2]) or
         iscolinear(x2[:, 0], x2[:, 1], x2[:, 3]) or
         iscolinear(x2[:, 0], x2[:, 2], x2[:, 3]) or
         iscolinear(x2[:, 1], x2[:, 2], x2[:, 3]))

    return r