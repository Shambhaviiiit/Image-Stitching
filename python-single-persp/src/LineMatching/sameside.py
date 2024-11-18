import numpy as np

def sameside(line, point1, point2):
    ss = False
    if np.isnan(point2[0]) or np.isnan(point2[1]):
        return ss

    if line['k'] != float('inf'):
        s1 = line['k'] * point1[0] + line['b'] - point1[1]
        s2 = line['k'] * point2[0] + line['b'] - point2[1]
    else:
        s1 = point1[0] - line['point1'][0]
        s2 = point2[0] - line['point1'][0]

    if s1 * s2 > 0:
        ss = True

    return ss
