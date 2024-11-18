import numpy as np
from src.modelspecific.vgg_get_homg import vgg_get_homg
from src.modelspecific.vgg_get_non_homg import vgg_get_nonhomg

def vgg_condition_2d(p, C):
    """
    Condition a set of 2D homogeneous or non-homogeneous points using conditioner C.
    
    Args:
    p : numpy array of shape (2, K) for non-homogeneous points, or (3, K) for homogeneous points.
    C : Conditioning matrix of shape (3, 3).
    
    Returns:
    pc : Conditionally transformed points.
    """
    r, c = p.shape

    if r == 2:
        # If points are non-homogeneous, convert them to homogeneous, apply the conditioner, and convert back to non-homogeneous
        pc = vgg_get_nonhomg(C @ vgg_get_homg(p))
    elif r == 3:
        # If points are already homogeneous, simply apply the conditioner
        pc = C @ p
    else:
        raise ValueError('Rows in points array must be either 2 or 3')

    return pc
