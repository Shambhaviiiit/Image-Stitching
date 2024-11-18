import numpy as np
from src.modelspecific.hnormalise import hnormalise


def homography_res(H, X):
    """
    Calculate the symmetric transfer error for a given homography and point correspondences.
    
    Arguments:
        H: 3x3 homography matrix.
        X: 6xnpts array of point correspondences, where:
            X[0:3,:] are points in the first image,
            X[3:6,:] are corresponding points in the second image.
    
    Returns:
        dist: Distances (errors) for each correspondence.
        H: Reshaped homography matrix.
    """
    # Reshape the homography matrix to 3x3
    # print("INSIDE RES")
    # print(H.shape)
    H = np.reshape(H, (3, 3))
    
    # Extract x1 and x2 (homogeneous coordinates)
    x1 = X[0:3, :]
    x2 = X[3:6, :]
    n = x1.shape[1]
    
    # Calculate the transferred points in both directions
    Hx1 = np.dot(H, x1)
    invHx2 = np.linalg.solve(H, x2)
    
    # Normalize the points so that the homogeneous scale is 1
    x1 = hnormalise(x1)
    x2 = hnormalise(x2)
    Hx1 = hnormalise(Hx1)
    invHx2 = hnormalise(invHx2)

    # print("X1")
    # print(x1.shape)
    # print(x2.shape)
    # print(Hx1.shape)
    # print(invHx2.shape)
    
    # Calculate the symmetric transfer error (sum of squared differences)
    a1 = (x1 - invHx2)**2
    a2 = (x2 - Hx1)**2
    print(a1.shape)
    print(a2.shape)
    # dist = np.sum(, axis=1) + np.sum(, axis = 1)
    # dist1 = np.sum((x1 - invHx2)**2)
    # dist2 = np.sum((x2 - Hx1)**2, axis = 0)
    # print(dist.shape)
    # print(dist1.shape)
    # print(dist2.shape)
    dist = np.sum((a1), axis=0) + np.sum((a2), axis=0)
    print("sum done")
    # Reshape the distance to a column vector
    dist = np.reshape(dist, (n, 1))
    
    # Return the distance and reshaped homography
    return dist, H.flatten()

