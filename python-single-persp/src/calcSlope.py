import numpy as np

def calc_slope(init_H, k, pts):
    # Calculate slope of original lines of slope k after transformation (through pts)
    h = init_H.T
    x, y = pts[0], pts[1]
    
    if not np.isinf(abs(k)):
        numerator = (k * x * h[3] * h[7] - k * x * h[4] * h[6] + k * h[5] * h[7] -
                     y * h[3] * h[7] + y * h[4] * h[6] - k * h[4] + h[5] * h[6] - h[3])
        denominator = (k * x * h[0] * h[7] - k * x * h[1] * h[6] + k * h[2] * h[7] -
                       y * h[0] * h[7] + y * h[1] * h[6] - k * h[1] + h[2] * h[6] - h[0])
    else:
        numerator = (x * h[3] * h[7] - x * h[4] * h[6] + h[5] * h[7] - h[4])
        denominator = (x * h[0] * h[7] - x * h[1] * h[6] + h[2] * h[7] - h[1])
    
    slope = numerator / denominator
    return slope
