import numpy as np
import cv2
from src.LineMatching.lineGradient import linegradient
from src.LineMatching.intspoints import intspoints

def paras(I, points):
    # Convert image to grayscale if it's not already
    if len(I.shape) > 2:  # If image has multiple channels (color image)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    # Get image size
    imsize = I.shape
    
    # Initialize the list of lines
    llines = []
    
    # Process each line
    for point in points:
        line = {}
        line['point1'] = [point[0], point[1]]
        line['point2'] = [point[2], point[3]]

        # Calculate line parameters (slope and intercept)
        if line['point2'][0] != line['point1'][0]:
            line['k'] = (line['point2'][1] - line['point1'][1]) / (line['point2'][0] - line['point1'][0])
            line['b'] = line['point1'][1] - line['k'] * line['point1'][0]
        else:
            line['k'] = float('inf')  # Infinite slope (vertical line)
            line['b'] = float('inf')  # Undefined intercept for vertical line

        llines.append(line)

    # Get the interpolated points for the lines
    pointlist = intspoints(llines, imsize)
    
    # Apply gradient calculations for lines
    llines = linegradient(I, llines)

    return llines, pointlist