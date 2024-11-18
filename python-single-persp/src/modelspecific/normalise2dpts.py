import numpy as np

def normalise2dpts(pts):
    """
    Normalises 2D homogeneous points so that their centroid is at the origin
    and their mean distance from the origin is sqrt(2). This improves the
    conditioning for solving homographies, fundamental matrices, etc.

    Parameters:
    pts (numpy.ndarray): 3xN array of 2D homogeneous coordinates

    Returns:
    newpts (numpy.ndarray): 3xN array of transformed 2D homogeneous coordinates
    T (numpy.ndarray): 3x3 transformation matrix, newpts = T @ pts
    """
    if pts.shape[0] != 3:
        raise ValueError("pts must be a 3xN array")
    
    # Find the indices of the points that are not at infinity
    finiteind = np.where(np.abs(pts[2, :]) > np.finfo(float).eps)[0]
    
    if len(finiteind) != pts.shape[1]:
        print('Some points are at infinity')
    
    # For the finite points ensure homogeneous coordinates have a scale of 1
    pts[0, finiteind] /= pts[2, finiteind]
    pts[1, finiteind] /= pts[2, finiteind]
    pts[2, finiteind] = 1
    
    # Compute the centroid of the finite points
    c = np.mean(pts[0:2, finiteind], axis=1)
    
    # Shift the origin to the centroid
    newp = np.zeros_like(pts)
    newp[0, finiteind] = pts[0, finiteind] - c[0]
    newp[1, finiteind] = pts[1, finiteind] - c[1]
    newp[2, finiteind] = 1  # Preserve homogeneous coordinate
    
    # Calculate the mean distance from the origin
    dist = np.sqrt(newp[0, finiteind]**2 + newp[1, finiteind]**2)
    meandist = np.mean(dist)
    
    # Scale factor to make the mean distance sqrt(2)
    scale = np.sqrt(2) / meandist
    
    # Transformation matrix
    T = np.array([[scale, 0, -scale * c[0]],
                  [0, scale, -scale * c[1]],
                  [0, 0, 1]])
    
    # Apply transformation to points
    newpts = T @ pts
    
    return newpts, T
