import numpy as np
import cv2
from src.modelspecific.normalise2dpts import normalise2dpts

def calcHomo(pts1, pts2):
    # Normalise point distribution
    data_pts = np.vstack([pts1, np.ones((1, pts1.shape[1])), pts2, np.ones((1, pts2.shape[1]))])
    
    # Normalize the points
    dat_norm_pts1, T1 = normalise2dpts(data_pts[:3, :])
    dat_norm_pts2, T2 = normalise2dpts(data_pts[3:, :])
    
    # Combine normalized points
    data_norm = np.vstack([dat_norm_pts1, dat_norm_pts2])
    
    # Use Direct Linear Transform (DLT) to compute the homography matrix
    h, _ = cv2.findHomography(dat_norm_pts1[:2, :].T, dat_norm_pts2[:2, :].T, method=0)
    
    # Reshape and compute the final homography
    H = np.linalg.inv(T2) @ (h.reshape(3, 3) @ T1)
    
    return H