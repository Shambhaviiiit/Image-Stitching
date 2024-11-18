import numpy as np
import cv2

# Ye GPT wala hai, actual wale mei C implemented hai, abhi tak check nahi kiya

def texture_mapping_ltl(img, ch, cw, C1, C2, X, Y, wX, wY, off):
    """
    Maps the warped image to the original image using texture mapping based on control vertices.
    
    :param img: The original image (numpy array)
    :param ch: Height of the warped image
    :param cw: Width of the warped image
    :param C1: Number of control points in the first dimension (height)
    :param C2: Number of control points in the second dimension (width)
    :param X: Original control points (X coordinates)
    :param Y: Original control points (Y coordinates)
    :param wX: Warped control points (X coordinates)
    :param wY: Warped control points (Y coordinates)
    :param off: Offset to adjust coordinates
    :return: The warped image as a numpy array
    """
    # Initialize an empty image to store the warped result
    warped_img = np.zeros((ch, cw, 3), dtype=np.uint8)
    
    # Apply the offset to the warped control points
    wX = wX - off[0]
    wY = wY - off[1]
    
    # Bilinear interpolation for mapping the texture to the original image
    for i in range(ch):
        for j in range(cw):
            # Calculate the corresponding (x, y) in the original image
            u = (i / (ch - 1)) * (C1 - 1)
            v = (j / (cw - 1)) * (C2 - 1)
            
            # Find the indices of the surrounding control points
            u0 = int(np.floor(u))
            u1 = min(u0 + 1, C1 - 1)
            v0 = int(np.floor(v))
            v1 = min(v0 + 1, C2 - 1)
            
            # Calculate the interpolation weights
            alpha_u = u - u0
            alpha_v = v - v0
            
            # Compute the corresponding pixel coordinates in the warped image
            src_x = (1 - alpha_u) * wX[u0, v0] + alpha_u * wX[u1, v0]
            src_y = (1 - alpha_u) * wY[u0, v0] + alpha_u * wY[u1, v0]
            
            # Bilinear interpolation on the image
            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                src_x = int(src_x)
                src_y = int(src_y)
                warped_img[i, j] = img[src_y, src_x]
    
    return warped_img

