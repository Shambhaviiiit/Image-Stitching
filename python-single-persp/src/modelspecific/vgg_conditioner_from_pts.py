import numpy as np
from src.modelspecific.vgg_get_non_homg import vgg_get_nonhomg

def vgg_conditioner_from_pts(Pts, isotropic=False):
    """
    Returns a conditioning matrix that normalizes points to have mean 0 and stddev sqrt(2).
    
    Args:
    Pts : numpy array of shape (D, K) where D is the dimensionality and K is the number of points.
    isotropic : boolean flag, if True the matrix will normalize in an isotropic way (i.e., all scale factors are the same).
    
    Returns:
    T : Conditioning matrix of shape (D, D).
    """
    Dim = Pts.shape[0]

    # Get non-homogeneous points (removes the last row if homogeneous)
    Pts = vgg_get_nonhomg(Pts)
    Pts = Pts[:Dim-1, :]  # Remove the last row for non-homogeneous points

    # Compute mean and standard deviation
    m = np.mean(Pts, axis=1)
    s = np.std(Pts, axis=1)
    s = s + (s == 0)  # To avoid division by zero

    if isotropic:
        # Isotropic case: Use the mean of all standard deviations for scaling
        mean_s = np.mean(s)
        T = np.vstack([np.diag(np.sqrt(2) / np.full(Dim-1, mean_s)), -np.dot(np.diag(np.sqrt(2) / s), m)])
    else:
        # Non-isotropic case: Scale each dimension individually
        T = np.vstack([np.diag(np.sqrt(2) / s), -np.dot(np.diag(np.sqrt(2) / s), m)])

    # Add the last column of zeros, ensuring the shape matches for the concatenation
    T = np.hstack([T, np.zeros((Dim, 1))])  # Add the last column of zeros
    
    # Add the last row [0...0 1], ensuring that T has the correct dimensions
    T = np.vstack([T, np.zeros(Dim)])  # Add the last row
    T[Dim-1, Dim-1] = 1  # Set the last element to 1

    return T[:3, :3]
