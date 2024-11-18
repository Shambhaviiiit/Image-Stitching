import math
import numpy as np

def Itspoint(line1, line2):
    """
    Function to calculate the intersection point of two lines.
    Args:
        line1: Dictionary containing 'k' (slope) and 'b' (intercept) for line 1.
        line2: Dictionary containing 'k' (slope) and 'b' (intercept) for line 2.
    Returns:
        X, Y: The intersection point (x, y) if lines are not parallel; NaN if lines are parallel.
    """
    k1 = line1['k']
    k2 = line2['k']
    b1 = line1['b']
    b2 = line2['b']
    
    X, Y = np.nan, np.nan
    
    if k1 == k2:
        # Lines are parallel (or coincident), no intersection
        return X, Y
    elif k1 != float('inf') and k2 != float('inf'):
        # Both lines have finite slopes
        if abs(math.atan((k2 - k1) / (1 + k1 * k2))) > math.pi / 8:
            X = (b2 - b1) / (k1 - k2)
            Y = k1 * X + b1
    elif k1 == float('inf'):
        # Line 1 is vertical (k1 is infinite)
        if abs(k2) <= 2.4142:
            X = line1['point1'][0]
            Y = k2 * X + b2
    elif k2 == float('inf'):
        # Line 2 is vertical (k2 is infinite)
        if abs(k1) <= 2.4142:
            X = line2['point1'][0]
            Y = k1 * X + b1

    return X, Y
