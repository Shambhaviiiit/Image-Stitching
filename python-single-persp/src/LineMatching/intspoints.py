import numpy as np
from src.LineMatching.Itspoint import Itspoint

def intspoints(lines, imsize):
    """
    Function to calculate intersection points between lines and store them in pointlist.
    Args:
        lines: List of lines, each line represented as a dictionary containing `point1` and `point2`.
        imsize: Tuple representing the image size (height, width).
    Returns:
        pointlist: List of points, each containing `point` (intersection) and `lines` (line indices).
    """
    len_lines = len(lines)
    pointlist = []
    
    for i in range(len_lines):
        for j in range(i+1, len_lines):
            # Get intersection point for lines[i] and lines[j]
            a, b = Itspoint(lines[i], lines[j])
            
            # Check if the intersection is within the image boundaries
            if 0 < a < imsize[1] and 0 < b < imsize[0]:
                
                # Check if the intersection is near the lines
                if isneighb([a, b], lines[i], lines[j]):
                    pointlist.append({'point': [a, b], 'lines': [i, j]})
    
    return pointlist

def isneighb(intp, line1, line2):
    """
    Function to check if an intersection point is near the endpoints of the given lines.
    Args:
        intp: Intersection point (x, y).
        line1: First line represented as a dictionary containing `point1` and `point2`.
        line2: Second line represented as a dictionary containing `point1` and `point2`.
    Returns:
        isneb: Boolean indicating if the intersection point is near the endpoints of the lines.
    """
    isneb = False
    
    if np.isnan(intp[0]) or np.isnan(intp[1]):
        return isneb
    
    # Calculate the length of the lines
    l1 = np.linalg.norm(np.array(line1['point1']) - np.array(line1['point2']))
    l2 = np.linalg.norm(np.array(line2['point1']) - np.array(line2['point2']))
    
    # Check if the intersection is near the endpoints of the lines
    if (np.linalg.norm(np.array(intp) - np.array(line1['point1'])) < 0.2 * l1 or
        np.linalg.norm(np.array(intp) - np.array(line1['point2'])) < 0.2 * l1) or \
       (np.linalg.norm(np.array(intp) - np.array(line2['point1'])) < 0.2 * l2 or
        np.linalg.norm(np.array(intp) - np.array(line2['point2'])) < 0.2 * l2):
        isneb = True
    
    return isneb

