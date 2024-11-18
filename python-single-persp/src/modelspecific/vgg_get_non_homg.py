import numpy as np

def vgg_get_nonhomg(x):
    """
    Convert a set of homogeneous points to non-homogeneous form.
    
    Args:
    x : numpy array of shape (D, K) where D is the dimensionality and K is the number of points.
    
    Returns:
    numpy array of shape (D-1, K) representing non-homogeneous points.
    """
    if x.size == 0:
        return np.array([])

    d = x.shape[0] - 1
    # Divide each coordinate by the last row (homogeneous coordinates)
    return x[:d, :] / x[-1, :]


