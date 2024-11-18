import numpy as np

def iscolinear(p1, p2, p3, flag='inhomog'):
    """
    Check if three points are collinear.

    Arguments:
        p1, p2, p3: Points in 2D or 3D.
        flag: Optional parameter set to 'h' or 'homog' indicating that 
              p1, p2, p3 are homogeneous coordinates with arbitrary scale.
              If omitted, it is assumed that the points are inhomogeneous.

    Returns:
        r: True if points are collinear, False otherwise.
    """
    
    if p1.shape != p2.shape or p1.shape != p3.shape or len(p1) not in [2, 3]:
        raise ValueError('Points must have the same dimension of 2 or 3')
    
    # If data is 2D, assume they are 2D inhomogeneous coords. Make them homogeneous with scale 1.
    if len(p1) == 2:
        p1 = np.append(p1, 1)
        p2 = np.append(p2, 1)
        p3 = np.append(p3, 1)
    
    if flag[0] == 'h':
        # Homogeneous coordinates with arbitrary scale.
        # Test that p1 X p2 generates a normal vector to the plane defined by p1 and p2.
        # If the dot product of this normal with p3 is zero, then p3 lies in the plane, hence collinear.
        r = np.abs(np.dot(np.cross(p1, p2), p3)) < 1e-6
    else:
        # Inhomogeneous coordinates or homogeneous with equal scale.
        r = np.linalg.norm(np.cross(p2 - p1, p3 - p1)) < 1e-6
    
    return r
