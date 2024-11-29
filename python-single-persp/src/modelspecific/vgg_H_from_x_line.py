import numpy as np
from src.modelspecific.vgg_conditioner_from_pts import vgg_conditioner_from_pts
from src.modelspecific.vgg_condition2d import vgg_condition_2d

def vgg_H_from_x_lin(xs1, xs2, A=None, W=None, C1=None, C2=None):
    # Check for size mismatch
    if xs1.shape != xs2.shape:
        raise ValueError('Input point sets are different sizes!')

    # Convert to homogeneous coordinates if necessary
    if xs1.shape[0] == 2:
        xs1 = np.vstack([xs1, np.ones(xs1.shape[1])])
        xs2 = np.vstack([xs2, np.ones(xs2.shape[1])])

    # Condition points if necessary
    if C1 is None or C2 is None:
        C1 = vgg_conditioner_from_pts(xs1)
        C2 = vgg_conditioner_from_pts(xs2)
        xs1 = vgg_condition_2d(xs1, C1)
        xs2 = vgg_condition_2d(xs2, C2)

    if W is not None and A is not None:
        # If W and A are provided, modify A with W
        B = A.copy()
        B[::2, :] = W @ A[::2, :]  # For odd rows
        B[1::2, :] = W @ A[1::2, :]  # For even rows

        # Perform SVD on the modified matrix
        u, s, v = np.linalg.svd(B)
    else:
        # Initialize A for point pair processing
        A = []
        ooo = np.zeros(3)
        
        # Loop over each point pair and add rows to A
        for k in range(xs1.shape[1]):  # xs1.shape[1] is the number of points
            p1 = xs1[:, k]
            p2 = xs2[:, k]
            
            # Add two rows per point pair to A
            A.append([p1[0] * p2[2], 0, -p1[0] * p2[0], p1[1] * p2[2], 0, -p1[1] * p2[0], p1[2] * p2[2], 0, -p1[2] * p2[0]])
            A.append([0, p1[0] * p2[2], -p1[0] * p2[1], 0, p1[1] * p2[2], -p1[1] * p2[1], 0, p1[2] * p2[2], -p1[2] * p2[1]])
        
        A = np.array(A)

        # Perform SVD on A
        u, s, v = np.linalg.svd(A)

    # Check for the null space dimension
    nullspace_dimension = np.sum(s < np.finfo(float).eps * s[0] * 1e3)
    if nullspace_dimension > 1:
        print('Nullspace is a bit roomy...')

    # Extract the last column of V (corresponding to the null space)
    h = v[:, 8]

    # Reshape h into a 3x3 homography matrix
    H = h.reshape(3, 3)

    # Decondition the homography
    H = np.linalg.inv(C2) @ H @ C1

    # Normalize the homography matrix
    H /= H[2, 2]

    return H, A, C1, C2