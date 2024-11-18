import numpy as np
import cv2

def pointgrad(I, i, j):
    # Ensure the indices are within valid bounds
    # print("ENTERED POINT GRAD")
    i = int(np.floor(i))
    j = int(np.floor(j))
    
    s = I.shape
    # if j < 1 or j >= s[0] or i < 1 or i >= s[1]:
    if j < 1 or j >= s[0] - 1 or i < 1 or i >= s[1] - 1:

        dx = 0
        dy = 0
    else:
        # Calculate gradients in x and y direction using central difference
        dx = (I[j, i+1] - I[j, i-1]) / 2.0
        dy = (I[j+1, i] - I[j-1, i]) / 2.0
    return dx, dy

def linegradient(I, lines):
    I = np.double(I)  # Convert image to double precision

    for i in range(len(lines)):
        ldx, ldy = 0, 0
        p1 = lines[i]['point1']
        p2 = lines[i]['point2']

        k1 = lines[i]['k']
        b1 = lines[i]['b']

        if -1 < k1 < 1:  # If the line is not vertical
            for ii in range(int(np.floor(min(p1[0], p2[0]))), int(np.ceil(max(p1[0], p2[0])))+1):
                jj = round(k1 * ii + b1)
                dx, dy = pointgrad(I, ii, jj)
                ldx += dx
                ldy += dy
        else:  # If the line is vertical
            for jj in range(int(np.floor(min(p1[1], p2[1]))), int(np.ceil(max(p1[1], p2[1])))+1):
                if k1 != float('inf'):
                    ii = round((jj - b1) / k1)
                else:
                    ii = p1[0]
                dx, dy = pointgrad(I, ii, jj)
                ldx += dx
                ldy += dy

        # Normalize the gradient and scale it
        gradient = np.array([ldx, ldy])
        norm = np.max(np.abs(gradient))
        if norm != 0:
            gradient = gradient / norm * 15  # Scale the gradient to 15 for visualization
        lines[i]['gradient'] = gradient

    return lines