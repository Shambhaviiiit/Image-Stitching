import numpy as np
import cv2

def refine_line(lines, img):
    sz1, sz2 = img.shape[:2]
    refine_lines = np.copy(lines)
    
    for i in range(lines.shape[0]):
        x1, y1, x2, y2, _, _, _ = lines[i]
        
        # Check if the line is within image bounds
        if 0 <= x1 < sz2 and 0 <= x2 < sz2 and 0 <= y1 < sz1 and 0 <= y2 < sz1:
            continue
        
        # Refine the line
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        if abs(b) <= np.finfo(float).eps:  # x1 == x2
            x1 = min(max(1.1, x1), sz2)
            x2 = min(max(1.1, x2), sz2)
            y1 = min(max(1.1, y1), sz1)
            y2 = min(max(1.1, y2), sz1)
        else:
            x1 = min(max(1.1, x1), sz2)
            x2 = min(max(1.1, x2), sz2)
            y1 = -(a * x1 + c) / b
            y2 = -(a * x2 + c) / b
        
        if abs(a) > np.finfo(float).eps:
            y1 = min(max(1.1, y1), sz1)
            y2 = min(max(1.1, y2), sz1)
            x1 = -(b * y1 + c) / a
            x2 = -(b * y2 + c) / a
        
        refine_lines[i, 0] = x1
        refine_lines[i, 1] = y1
        refine_lines[i, 2] = x2
        refine_lines[i, 3] = y2
    
    return refine_lines
