import numpy as np
import cv2

def generate_uv(img1, img2, init_H, theta, C1, C2):
    # Given an orientation of img, rotate the original control vertices Mv
    # Obtain the orthogonal equidistant lines
    # Generating v-slope lines, u-slope lines, u-equal lines
    # Based on u sample points in non-overlapping region

    multi = 2
    M, N, _ = img1.shape  # size of img
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # rotation matrix
    off_center = np.array([(N+1)/2, (M+1)/2]) - R @ np.array([(N+1)/2, (M+1)/2])  # offset of rotated center wrt original center

    # Roughly calculate the overlapping boundary to filter u-sample points
    warp_pts1 = init_H @ np.array([1, 1, 1])
    warp_pts1 /= warp_pts1[2]
    left_or_right = warp_pts1[0] <= 1
    sz1, sz2, _ = img2.shape

    if left_or_right:  # left is target, line x=1 is boundary
        inv_pts1 = np.linalg.inv(init_H) @ np.array([1, -sz1, 1])
        inv_pts2 = np.linalg.inv(init_H) @ np.array([1, 2 * sz1, 1])
    else:  # right is target, line x=N is boundary
        inv_pts1 = np.linalg.inv(init_H) @ np.array([sz2, -sz1, 1])
        inv_pts2 = np.linalg.inv(init_H) @ np.array([sz2, 2 * sz1, 1])

    inv_pts1 /= inv_pts1[2]
    inv_pts2 /= inv_pts2[2]
    x1, y1 = inv_pts1[:2]
    x2, y2 = inv_pts2[:2]

    # Generating mesh grid (C1*C2) to optimize warped control vertices
    X, Y = np.meshgrid(np.linspace(2-N, 2*N-1, 3*multi*(C2+1)-2),
                       np.linspace(2-M, 2*M-1, 3*multi*(C1+1)-2))
    
    lines_vs = np.zeros((2 * X.shape[1], X.shape[0]))
    k = 0

    # Rotated vertical line slope-preserving
    for j in range(X.shape[1]):
        tmp_Mv = np.vstack([X[:, j], Y[:, j]])  # each vertical line
        tmp_line = R @ tmp_Mv + off_center[:, np.newaxis]  # rotated vertical line
        inner_ind = (tmp_line[0, :] >= 1) & (tmp_line[0, :] <= N) & (tmp_line[1, :] >= 1) & (tmp_line[1, :] <= M)
        lines_vs[2*k:2*k+2, :np.sum(inner_ind)] = tmp_line[:, inner_ind]  # useful rotated vertical line
        lines_vs[2*k:2*k+2, -1] = np.sum(inner_ind)  # number of sample points on the line
        k += 1

    # Rotated horizontal line slope-preserving
    lines_us = np.zeros((2 * X.shape[0], X.shape[1]))
    k = 0

    for i in range(X.shape[0]):
        tmp_Mv = np.vstack([X[i, :], Y[i, :]])  # each horizontal line
        tmp_line = R @ tmp_Mv + off_center[:, np.newaxis]  # rotated horizontal line
        inner_ind = (tmp_line[0, :] >= 1) & (tmp_line[0, :] <= N) & (tmp_line[1, :] >= 1) & (tmp_line[1, :] <= M)
        lines_us[2*k:2*k+2, :np.sum(inner_ind)] = tmp_line[:, inner_ind]  # useful rotated horizontal line
        lines_us[2*k:2*k+2, -1] = np.sum(inner_ind)  # number of sample points on the line
        k += 1

    # Remove all-zero rows and columns
    lines_vs = lines_vs[~np.all(lines_vs == 0, axis=1)]
    lines_us = lines_us[~np.all(lines_us == 0, axis=1)]
    lines_vs = lines_vs[:, ~np.all(lines_vs == 0, axis=0)]
    lines_us = lines_us[:, ~np.all(lines_us == 0, axis=0)]

    # Filter u sample points (omit points in the overlapping region)
    newlines_u = np.zeros_like(lines_us)
    for i in range(0, lines_us.shape[0] - 1, 2):
        num_u = int(lines_us[i, -1])
        vec_prod = (x1 - lines_us[i, :num_u]) * (y2 - lines_us[i+1, :num_u]) + \
                   (lines_us[i+1, :num_u] - y1) * (x2 - lines_us[i, :num_u])
        left = vec_prod > 0
        right = vec_prod < 0
        index_prod = (left_or_right & left) | ((~left_or_right) & right)  # sample points in non-overlapping region
        newlines_u[i:i+2, :np.sum(index_prod)] = lines_us[i:i+2, index_prod]
        newlines_u[i:i+2, -1] = np.sum(index_prod)  # number of sample points

    lines_ue = newlines_u
    lines_ue = lines_ue[~np.all(lines_ue == 0, axis=1)]
    lines_ue = lines_ue[:, ~np.all(lines_ue == 0, axis=0)]

    return lines_vs, lines_us, lines_ue
