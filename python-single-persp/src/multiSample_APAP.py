import numpy as np
from src.modelspecific.normalise2dpts import normalise2dpts
from src.multigs.multigsSampling import multigs_sampling

def multi_sample_apap(pts1, pts2):
    # global fitfn, resfn, degenfn, psize, numpar
    # fitfn = 'homography_fit'
    # resfn = 'homography_res'
    # degenfn = 'homography_degen'
    # psize = 4
    # numpar = 9
    print("PTS1")
    print(pts1.shape)
    # Normalize point distribution
    # data_orig = np.vstack([pts1[:2, :], np.ones((1, pts1.shape[1])), pts2[:2, :], np.ones((1, pts2.shape[1]))])
    data_orig = np.vstack([pts1.T, np.ones((1, pts1.shape[0])), pts2.T, np.ones((1, pts2.shape[0]))])
    data_norm_img1, _ = normalise2dpts(data_orig[:3, :])
    data_norm_img2, _ = normalise2dpts(data_orig[3:, :])
    data_norm = np.vstack([data_norm_img1, data_norm_img2])
    
    print("DATA NORM")
    print(data_norm.shape)
    # Outlier removal - Multi-GS (RANSAC)
    np.random.seed(0)
    _, res, _, _, _ = multigs_sampling(100, data_norm, 500, 10)
    print("RES")
    print(res)
    con = np.sum(res <= 0.05, axis=0)
    maxinx = np.argmax(con)
    inliers = np.where(res[:, maxinx] <= 0.05)[0]
    print("INLIERS")
    print(inliers)

    matches_1 = pts1[:, inliers]
    matches_2 = pts2[:, inliers]
    
    # Remove duplicate feature matches
    matches_1, unique_indices_1 = np.unique(matches_1, axis=1, return_index=True)
    matches_2 = matches_2[:, unique_indices_1]
    matches_2, unique_indices_2 = np.unique(matches_2, axis=1, return_index=True)
    matches_1 = matches_1[:, unique_indices_2]
    
    return matches_1, matches_2