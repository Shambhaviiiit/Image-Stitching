import numpy as np
from src.texture_mapping.texture_mapping_ltl import texture_mapping_ltl

def meshmap_warp2homo(img, X, Y, wX, wY):
    # Given original control vertices X, Y and warped control vertices wX, wY,
    # map warped image to original image using mesh-homography.
    
    # Calculate offsets
    off = np.round([1 - np.min(wX), 1 - np.min(wY)])
    
    # Calculate the size of the warped image
    cw = np.round(np.max(wX) - np.min(wX)) + 1
    ch = np.round(np.max(wY) - np.min(wY)) + 1
    
    # Get the number of control points in X and Y
    C1 = X.shape[0] - 1
    C2 = X.shape[1] - 1
    
    # Perform texture mapping (assuming you have implemented texture_mapping_ltl in Python)
    warped_img = texture_mapping_ltl(img, ch, cw, C1, C2, X, Y, wX, wY, off)
    
    # Reshape the warped image to include RGB channels
    warped_img = np.reshape(warped_img, (warped_img.shape[0], warped_img.shape[1] // 3, 3))
    
    return warped_img

