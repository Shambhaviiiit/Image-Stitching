import numpy as np
from src.calcHomo import calcHomo

def linesDelete(line1, line2, pts1, pts2):
    outlier_threshold = 3  # Outlier threshold for projection distance
    print(pts1.shape)
    # Delete duplicate feature matches
    
    # pts1: (n,2)
    # matches1: (2, n')
    indh1 = np.lexsort((pts1[:, 1], pts1[:, 0]))  
    h1 = pts1[indh1]  # Apply sorting
    _, indk1 = np.unique(h1, axis=0, return_index=True)  # Find unique rows
    matches1 = pts1[indh1[indk1], :].T
    matches2 = pts2[indh1[indk1], :].T

    matches2_T = matches2.T
    indh2 = np.lexsort((matches2_T[:, 1], matches2_T[:, 0]))
    h2 = matches2_T[indh2]
    _, indk2 = np.unique(h2, axis=0, return_index=True)
    # matches1 = matches1[:, indk2]
    # matches2 = matches2[:, indk2]
    matches1 = matches1[:, indh2[indk2]]  # Filter matches1 based on matches2
    matches2 = matches2[:, indh2[indk2]]  # Filter matches2

    print("matches")
    print(matches1.shape)
    print(matches2.shape)
    
    # Calculate homography matrix
    init_H = calcHomo(matches1, matches2)
    num_line = line1.shape[0]
    
    print("line input")
    print(line1.shape)
    
    # Equation for line 1
    abc_line1 = np.array([
        line1[:, 3] - line1[:, 1], 
        line1[:, 0] - line1[:, 2], 
        line1[:, 2]*line1[:, 1] - line1[:, 0]*line1[:, 3]
    ]).T
    
    # Equation for line 2
    abc_line2 = np.array([
        line2[:, 3] - line2[:, 1], 
        line2[:, 0] - line2[:, 2], 
        line2[:, 2]*line2[:, 1] - line2[:, 0]*line2[:, 3]
    ]).T
    
    # Warp line 1 and calculate projection distance
    aux_line1 = np.vstack([line1[:, :2].T, np.ones((1, num_line)), line1[:, 2:4].T, np.ones((1, num_line))])
    warp_p1 = np.dot(init_H, aux_line1[:3, :])
    warp_p1 /= warp_p1[2, :]
    
    warp_p2 = np.dot(init_H, aux_line1[3:6, :])
    warp_p2 /= warp_p2[2, :]
    
    dist_p1 = np.abs(np.sum(warp_p1.T * abc_line2, axis=1)) / np.sqrt(abc_line2[:, 0]**2 + abc_line2[:, 1]**2)
    dist_p2 = np.abs(np.sum(warp_p2.T * abc_line2, axis=1)) / np.sqrt(abc_line2[:, 0]**2 + abc_line2[:, 1]**2)
    
    mean_dist = (dist_p1 + dist_p2) / 2
    inliers = mean_dist <= outlier_threshold
    
    # Warp line 2 and calculate projection distance
    aux_line2 = np.vstack([line2[:, :2].T, np.ones((1, num_line)), line2[:, 2:4].T, np.ones((1, num_line))])
    warp_p1_ = np.linalg.solve(init_H, aux_line2[:3, :])
    warp_p1_ /= warp_p1_[2, :]
    
    warp_p2_ = np.linalg.solve(init_H, aux_line2[3:6, :])
    warp_p2_ /= warp_p2_[2, :]
    
    dist_p1_ = np.abs(np.sum(warp_p1_.T * abc_line1, axis=1)) / np.sqrt(abc_line1[:, 0]**2 + abc_line1[:, 1]**2)
    dist_p2_ = np.abs(np.sum(warp_p2_.T * abc_line1, axis=1)) / np.sqrt(abc_line1[:, 0]**2 + abc_line1[:, 1]**2)
    
    mean_dist_ = (dist_p1_ + dist_p2_) / 2
    inliers_ = mean_dist_ <= outlier_threshold
    
    # Combine the inliers from both checks
    line_inliers = inliers & inliers_
    
    # Return inliers for both line1 and line2
    inlier_line1 = line1[line_inliers, :]
    inlier_line2 = line2[line_inliers, :]
    
    return inlier_line1, inlier_line2