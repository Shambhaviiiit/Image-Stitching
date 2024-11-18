import numpy as np

def projline(H, lines):
    n = len(lines)
    plines = []
    
    for i in range(n):
        line = lines[i]
        
        point1_proj = projpoint(H, line['point1'])
        point2_proj = projpoint(H, line['point2'])
        
        line_proj = {}
        line_proj['point1'] = point1_proj
        line_proj['point2'] = point2_proj
        
        if point2_proj[0] != point1_proj[0]:
            line_proj['k'] = (point2_proj[1] - point1_proj[1]) / (point2_proj[0] - point1_proj[0])
            line_proj['b'] = point1_proj[1] - line_proj['k'] * point1_proj[0]
        else:
            line_proj['k'] = float('inf')
            line_proj['b'] = float('inf')
        
        line_proj['ind'] = i
        plines.append(line_proj)
    
    return plines


def projpoint(H, point):
    # Convert point to homogeneous coordinates
    point_homogeneous = np.array([point[0], point[1], 1])
    
    # Apply the homography
    p = np.dot(H, point_homogeneous)
    
    # Convert back to Cartesian coordinates
    p = p / p[2]
    
    return p[:2]  # Return only the (x, y) part
