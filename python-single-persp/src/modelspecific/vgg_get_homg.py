import numpy as np

def vgg_get_homg(x):
    """
    Convert a set of non-homogeneous points to homogeneous form.
    
    Args:
    x : numpy array of shape (D-1, K) where D is the dimensionality and K is the number of points.
    
    Returns:
    numpy array of shape (D, K) representing homogeneous points.
    """
    ones = np.ones((1, x.shape[1]))
    return np.vstack([x, ones])
