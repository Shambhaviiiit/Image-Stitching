import numpy as np

def hnormalise(x):
    """
    Normalizes an array of homogeneous coordinates to a scale of 1.
    
    Arguments:
        x: An NxNpts array of homogeneous coordinates.
    
    Returns:
        nx: An NxNpts array of homogeneous coordinates rescaled such that the
            scale values nx(N,:) are all 1. Homogeneous coordinates at infinity
            (scale value of 0) are left unchanged.
    """
    rows, npts = x.shape
    nx = np.copy(x)

    # Find the indices of the points that are not at infinity (scale != 0)
    finiteind = np.abs(x[rows-1, :]) > np.finfo(float).eps

    if np.sum(finiteind) != npts:
        print('Some points are at infinity')

    # Normalize points that are not at infinity
    for r in range(rows - 1):
        nx[r, finiteind] = x[r, finiteind] / x[rows-1, finiteind]
    nx[rows-1, finiteind] = 1

    return nx
