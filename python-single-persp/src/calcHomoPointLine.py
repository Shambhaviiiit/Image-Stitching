import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
from src.modelspecific.normalise2dpts import normalise2dpts

# def normalise2dpts(pts):
#     """
#     Normalize a set of 2D homogeneous points.
#     """
#     finite_inds = np.nonzero(pts[2, :])[0]
#     pts[:, finite_inds] /= pts[2, finite_inds]
#     centroid = np.mean(pts[:2, finite_inds], axis=1)
#     centered_pts = pts[:2, finite_inds] - centroid[:, np.newaxis]
#     mean_dist = np.mean(np.sqrt(np.sum(centered_pts**2, axis=0)))
#     scale = np.sqrt(2) / mean_dist
#     T = np.array([[scale, 0, -scale * centroid[0]],
#                   [0, scale, -scale * centroid[1]],
#                   [0, 0, 1]])
#     norm_pts = np.dot(T, pts)
#     return norm_pts, T

def myfun(x, pts2, tmp_line2):
    """
    Least-square cost function for optimization.
    """
    a = tmp_line2[:, 0]
    b = tmp_line2[:, 1]
    c = tmp_line2[:, 2]

    term1 = np.mean(np.sqrt((x[0] * pts2[0, :] + x[1])**2 + (x[0] * pts2[1, :] + x[2])**2)) - np.sqrt(2)
    term2 = np.mean(np.abs(x[0]**2 * c - x[0] * x[1] * a - x[0] * x[2] * b) / np.sqrt((x[0] * a)**2 + (x[0] * b)**2)) - np.sqrt(1/2)
    
    return np.array([term1, term2])

def calc_homo_point_line(pts1_org, pts2_org, line1, line2):
    """
    Estimate homography based on feature matches and line segment matches.
    """
    pts1 = pts1_org.T
    pts2 = pts2_org.T
    num_pts = pts1.shape[1]
    num_line = line1.shape[0]
    print("line shape in calc homo")
    print(line1.shape)

    # Point-centric normalization before SVD w.r.t. dual-feature
    re_line1 = line1.T.reshape(2, 2 * num_line)
    norm_pts_line1, T1 = normalise2dpts(np.vstack([pts1, re_line1, np.ones((1, num_pts + 2 * num_line))]))
    norm_pts1 = norm_pts_line1[:2, :num_pts]
    norm_line1 = norm_pts_line1[:2, num_pts:].reshape(4, num_line)

    # Line function calculation for the reference image
    abc_line2 = np.hstack([
        (line2[:, 3] - line2[:, 1]).reshape(-1, 1),
        (line2[:, 0] - line2[:, 2]).reshape(-1, 1),
        (line2[:, 2] * line2[:, 1] - line2[:, 0] * line2[:, 3]).reshape(-1, 1)
    ])
    tmp_line2 = abc_line2 / abc_line2[:, [2]]

    # Least-squares method for normalization of target image
    x0 = np.array([1, 0, 0])
    res = least_squares(myfun, x0, args=(pts2, tmp_line2))
    x = res.x
    T2 = np.array([[x[0], 0, x[1]], [0, x[0], x[2]], [0, 0, 1]])

    norm_pts2 = T2 @ np.vstack([pts2, np.ones((1, num_pts))])
    norm_line2 = np.hstack([
        x[0] * abc_line2[:, [0]], x[0] * abc_line2[:, [1]],
        x[0]**2 * abc_line2[:, [2]] - x[0] * x[1] * abc_line2[:, [0]] - x[0] * x[2] * abc_line2[:, [1]]
    ])

    # Generate feature matches value matrix A (||Ah||=0)
    normA_pts = np.zeros((2 * num_pts, 9))
    x1 = norm_pts1[0, :].reshape(-1, 1)
    y1 = norm_pts1[1, :].reshape(-1, 1)
    x2 = norm_pts2[0, :].reshape(-1, 1)
    y2 = norm_pts2[1, :].reshape(-1, 1)

    normA_pts[0::2, 0:3] = np.hstack([x1, y1, np.ones((num_pts, 1))])
    normA_pts[0::2, 6:9] = -x2 * np.hstack([x1, y1, np.ones((num_pts, 1))])
    normA_pts[1::2, 3:6] = np.hstack([x1, y1, np.ones((num_pts, 1))])
    normA_pts[1::2, 6:9] = -y2 * np.hstack([x1, y1, np.ones((num_pts, 1))])

    # Generate line segments matches value matrix B (||Bh||=0)
    u0, v0 = norm_line1[:2, :]
    u1, v1 = norm_line1[2:, :]
    a2, b2, c2 = norm_line2.T
    k = 1.0 / (a2**2 + b2**2)

    normB_line = np.zeros((2 * num_line, 9))
    normB_line[0::2, :] = np.hstack([k * a2 * u0, k * a2 * v0, k * a2, k * b2 * u0, k * b2 * v0, k * b2, k * c2 * u0, k * c2 * v0, k * c2])
    normB_line[1::2, :] = np.hstack([k * a2 * u1, k * a2 * v1, k * a2, k * b2 * u1, k * b2 * v1, k * b2, k * c2 * u1, k * c2 * v1, k * c2])

    # Solve homography with ||[A;B]h=0||
    norm_C = np.vstack([normA_pts, normB_line])
    _, _, V = svd(norm_C)
    h = V[:, -1].reshape(3, 3)

    return h, norm_C, T1, T2
