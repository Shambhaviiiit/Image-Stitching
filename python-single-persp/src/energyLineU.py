import numpy as np
from scipy.sparse import coo_matrix
from src.calcSlope import calc_slope
from src.meshGridAlign import mesh_grid_align

def energy_line_u(img, C1, C2, lines_us, lines_ue, init_H):
    # for given target image, C1*C2 mesh grid and lines in target image,
    # generate sparse matrix for line-preserving
    # for lines_u, preserve slope-invariant
    num_V = (C1 + 1) * (C2 + 1)  # number of control vertices
    X_col = np.linspace(1, img.shape[1], C2 + 1)  # column index of cells
    Y_row = np.linspace(1, img.shape[0], C1 + 1)  # row index of cells
    x_dis = X_col[1] - X_col[0]  # the width of scale-cell
    y_dis = Y_row[1] - Y_row[0]  # the height of scale-cell

    Mesh_ps = np.zeros((4, 2))
    Mesh_pe = np.zeros((4, 2))
    Mesh_pm = np.zeros((4, 2))

    # rotated horizontal line, slope-preserving
    hor_sp = np.sum(lines_us[0::2, -1] - 1)
    sp_i = np.zeros(16 * hor_sp, dtype=int)  # row index
    sp_j = np.zeros(16 * hor_sp, dtype=int)  # column index
    sp_s = np.zeros(16 * hor_sp)  # value index
    k = 1
    init_k = init_H[6] / init_H[3]

    for i in range(0, len(lines_us) - 1, 2):
        num_u = lines_us[i, -1]
        if num_u <= 1:
            continue  # if sample points less than 2, continue
        k_xy = calc_slope(init_H, init_k, np.array([lines_us[i, 0], lines_us[i + 1, 0]]))  # slope of u-line after transformation
        if not np.isinf(np.abs(k_xy)):
            nor_vec = np.array([k_xy, -1])
        else:
            nor_vec = np.array([1, 0])
        nor_vec = nor_vec / np.linalg.norm(nor_vec)  # normal vector of warped u-lines
        for j in range(num_u - 1):
            lps = [lines_us[i, j], lines_us[i + 1, j]]
            lpe = [lines_us[i, j + 1], lines_us[i + 1, j + 1]]
            pxs = min(np.where(lps[0] - X_col < x_dis)[0][0], C2)  # the x index of p's position
            pys = min(np.where(lps[1] - Y_row < y_dis)[0][0], C1)  # the y index of p's position
            pxe = min(np.where(lpe[0] - X_col < x_dis)[0][0], C2)  # the x index of p's position
            pye = min(np.where(lpe[1] - Y_row < y_dis)[0][0], C1)  # the y index of p's position

            nums1 = (C1 + 1) * (pxs - 1) + pys  # index of v1*
            nums2 = nums1 + C1 + 1
            nums3 = nums2 + 1
            nums4 = nums1 + 1
            nume1 = (C1 + 1) * (pxe - 1) + pye
            nume2 = nume1 + C1 + 1
            nume3 = nume2 + 1
            nume4 = nume1 + 1

            Mesh_ps[0:4, :] = np.array([[X_col[pxs], Y_row[pys]],  # v1
                                         [X_col[pxs + 1], Y_row[pys]],  # v2
                                         [X_col[pxs + 1], Y_row[pys + 1]],  # v3
                                         [X_col[pxs], Y_row[pys + 1]]])  # v4
            Mesh_pe[0:4, :] = np.array([[X_col[pxe], Y_row[pye]],  # v1
                                         [X_col[pxe + 1], Y_row[pye]],  # v2
                                         [X_col[pxe + 1], Y_row[pye + 1]],  # v3
                                         [X_col[pxe], Y_row[pye + 1]]])  # v4

            coeff_mesh_ps = mesh_grid_align(Mesh_ps, lps)
            coeff_mesh_pe = mesh_grid_align(Mesh_pe, lpe)

            sp_i[16 * k - 16:16 * k] = k * np.ones(16, dtype=int)
            sp_j[16 * k - 16:16 * k] = np.array([2 * nums1 - 1, 2 * nums2 - 1, 2 * nums3 - 1, 2 * nums4 - 1,
                                                  2 * nume1 - 1, 2 * nume2 - 1, 2 * nume3 - 1, 2 * nume4 - 1,
                                                  2 * nums1, 2 * nums2, 2 * nums3, 2 * nums4,
                                                  2 * nume1, 2 * nume2, 2 * nume3, 2 * nume4])
            sp_s[16 * k - 16:16 * k] = np.array([-nor_vec[0] * coeff_mesh_ps, nor_vec[0] * coeff_mesh_pe,
                                                 -nor_vec[1] * coeff_mesh_ps, nor_vec[1] * coeff_mesh_pe])
            k += 1

    sparse_us = coo_matrix((sp_s, (sp_i, sp_j)), shape=(hor_sp, 2 * num_V))

    # rotated horizontal line, equidistant-preserving
    hor_sp = 2 * np.sum(np.maximum(0, lines_ue[0::2, -1] - 2))
    sp_i = np.zeros(12 * hor_sp, dtype=int)  # row index
    sp_j = np.zeros(12 * hor_sp, dtype=int)  # column index
    sp_s = np.zeros(12 * hor_sp)  # value index
    k = 1

    for i in range(0, len(lines_ue) - 1, 2):
        num_u = lines_ue[i, -1]
        if num_u <= 2:
            continue  # if sample points less than 2, continue
        for j in range(1, num_u - 1):
            lps = [lines_ue[i, j - 1], lines_ue[i + 1, j - 1]]
            lpm = [lines_ue[i, j], lines_ue[i + 1, j]]
            lpe = [lines_ue[i, j + 1], lines_ue[i + 1, j + 1]]
            pxs = min(np.where(lps[0] - X_col < x_dis)[0][0], C2)  # the x index of p's position
            pys = min(np.where(lps[1] - Y_row < y_dis)[0][0], C1)  # the y index of p's position
            pxm = min(np.where(lpm[0] - X_col < x_dis)[0][0], C2)  # the x index of p's position
            pym = min(np.where(lpm[1] - Y_row < y_dis)[0][0], C1)  # the y index of p's position
            pxe = min(np.where(lpe[0] - X_col < x_dis)[0][0], C2)  # the x index of p's position
            pye = min(np.where(lpe[1] - Y_row < y_dis)[0][0], C1)  # the y index of p's position

            nums1 = (C1 + 1) * (pxs - 1) + pys  # index of v1*
            nums2 = nums1 + C1 + 1
            nums3 = nums2 + 1
            nums4 = nums1 + 1
            numm1 = (C1 + 1) * (pxm - 1) + pym  # index of v1*
            numm2 = numm1 + C1 + 1
            numm3 = numm2 + 1
            numm4 = numm1 + 1
            nume1 = (C1 + 1) * (pxe - 1) + pye
            nume2 = nume1 + C1 + 1
            nume3 = nume2 + 1
            nume4 = nume1 + 1

            Mesh_ps[0:4, :] = np.array([[X_col[pxs], Y_row[pys]],  # v1
                                         [X_col[pxs + 1], Y_row[pys]],  # v2
                                         [X_col[pxs + 1], Y_row[pys + 1]],  # v3
                                         [X_col[pxs], Y_row[pys + 1]]])  # v4
            Mesh_pe[0:4, :] = np.array([[X_col[pxe], Y_row[pye]],  # v1
                                         [X_col[pxe + 1], Y_row[pye]],  # v2
                                         [X_col[pxe + 1], Y_row[pye + 1]],  # v3
                                         [X_col[pxe], Y_row[pye + 1]]])  # v4

            coeff_mesh_ps = mesh_grid_align(Mesh_ps, lps)
            coeff_mesh_pe = mesh_grid_align(Mesh_pe, lpe)

            sp_i[12 * k - 12:12 * k] = k * np.ones(12, dtype=int)
            sp_j[12 * k - 12:12 * k] = np.array([2 * nums1 - 1, 2 * nums2 - 1, 2 * nums3 - 1, 2 * nums4 - 1,
                                                  2 * numm1 - 1, 2 * numm2 - 1, 2 * numm3 - 1, 2 * numm4 - 1,
                                                  2 * nume1 - 1, 2 * nume2 - 1, 2 * nume3 - 1, 2 * nume4 - 1])
            sp_s[12 * k - 12:12 * k] = np.array([nor_vec[0] * coeff_mesh_ps, -nor_vec[0] * coeff_mesh_pe,
                                                 nor_vec[1] * coeff_mesh_ps, -nor_vec[1] * coeff_mesh_pe])
            k += 1

    sparse_ue = coo_matrix((sp_s, (sp_i, sp_j)), shape=(hor_sp, 2 * num_V))

    return sparse_us, sparse_ue
