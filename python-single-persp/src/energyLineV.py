import numpy as np
from scipy.sparse import csr_matrix
from src.meshGridAlign import mesh_grid_align

def energy_line_v(img, C1, C2, lines_v, nor_vec):
    """
    For a given target image, C1*C2 mesh grid and lines in target image,
    generate sparse matrix for line-preserving.
    For lines_v, preserve slope and equidistant properties.
    """
    num_V = (C1 + 1) * (C2 + 1)  # number of control vertices
    
    X_col = np.linspace(1, img.shape[1], C2 + 1)  # column index of cells
    Y_row = np.linspace(1, img.shape[0], C1 + 1)  # row index of cells
    x_dis = X_col[1] - X_col[0]  # width of scale-cell
    y_dis = Y_row[1] - Y_row[0]  # height of scale-cell
    
    Mesh_ps = np.zeros((4, 2))
    Mesh_pe = np.zeros((4, 2))
    Mesh_pm = np.zeros((4, 2))
    
    # Rotated vertical line, orthogonal with normal vector
    row_sp = np.sum(lines_v[::2, -1] - 1)
    sp_i = np.zeros(16 * row_sp)
    sp_j = np.zeros(16 * row_sp)
    sp_s = np.zeros(16 * row_sp)
    k = 0

    for i in range(0, lines_v.shape[0] - 1, 2):
        num_s = lines_v[i, -1]
        if num_s <= 1:
            continue
        
        for j in range(num_s - 1):
            lps = [lines_v[i, j], lines_v[i + 1, j]]
            lpm = [lines_v[i, j + 1], lines_v[i + 1, j + 1]]
            
            pxs = min(np.where(lps[0] - X_col < x_dis, 1)[0], C2)
            pys = min(np.where(lps[1] - Y_row < y_dis, 1)[0], C1)
            pxm = min(np.where(lpm[0] - X_col < x_dis, 1)[0], C2)
            pym = min(np.where(lpm[1] - Y_row < y_dis, 1)[0], C1)

            nums1 = (C1 + 1) * (pxs - 1) + pys
            nums2 = nums1 + C1 + 1
            nums3 = nums2 + 1
            nums4 = nums1 + 1
            numm1 = (C1 + 1) * (pxm - 1) + pym
            numm2 = numm1 + C1 + 1
            numm3 = numm2 + 1
            numm4 = numm1 + 1
            
            Mesh_ps[:4, :] = [[X_col[pxs], Y_row[pys]], [X_col[pxs + 1], Y_row[pys]],
                              [X_col[pxs + 1], Y_row[pys + 1]], [X_col[pxs], Y_row[pys + 1]]]
            Mesh_pm[:4, :] = [[X_col[pxm], Y_row[pym]], [X_col[pxm + 1], Y_row[pym]],
                              [X_col[pxm + 1], Y_row[pym + 1]], [X_col[pxm], Y_row[pym + 1]]]
            
            coeff_mesh_ps = mesh_grid_align(Mesh_ps, lps)
            coeff_mesh_pm = mesh_grid_align(Mesh_pm, lpm)

            sp_i[16 * k: 16 * (k + 1)] = k + 1
            sp_j[16 * k: 16 * (k + 1)] = [
                2 * nums1 - 1, 2 * nums2 - 1, 2 * nums3 - 1, 2 * nums4 - 1,
                2 * numm1 - 1, 2 * numm2 - 1, 2 * numm3 - 1, 2 * numm4 - 1,
                2 * nums1, 2 * nums2, 2 * nums3, 2 * nums4,
                2 * numm1, 2 * numm2, 2 * numm3, 2 * numm4
            ]
            sp_s[16 * k: 16 * (k + 1)] = np.concatenate([
                -nor_vec[0] * coeff_mesh_ps, nor_vec[0] * coeff_mesh_pm,
                -nor_vec[1] * coeff_mesh_ps, nor_vec[1] * coeff_mesh_pm
            ])
            k += 1

    # Rotated vertical line, equidistance on sample points
    row_spp = 2 * np.sum(np.maximum(0, lines_v[::2, -1] - 2))
    sp_ii = np.zeros(12 * row_spp)
    sp_jj = np.zeros(12 * row_spp)
    sp_ss = np.zeros(12 * row_spp)
    k = 0

    for i in range(0, lines_v.shape[0] - 1, 2):
        num_s = lines_v[i, -1]
        if num_s <= 2:
            continue
        
        for j in range(1, num_s - 1):
            lps = [lines_v[i, j - 1], lines_v[i + 1, j - 1]]
            lpm = [lines_v[i, j], lines_v[i + 1, j]]
            lpe = [lines_v[i, j + 1], lines_v[i + 1, j + 1]]

            pxs = min(np.where(lps[0] - X_col < x_dis, 1)[0], C2)
            pys = min(np.where(lps[1] - Y_row < y_dis, 1)[0], C1)
            pxm = min(np.where(lpm[0] - X_col < x_dis, 1)[0], C2)
            pym = min(np.where(lpm[1] - Y_row < y_dis, 1)[0], C1)
            pxe = min(np.where(lpe[0] - X_col < x_dis, 1)[0], C2)
            pye = min(np.where(lpe[1] - Y_row < y_dis, 1)[0], C1)

            nums1 = (C1 + 1) * (pxs - 1) + pys
            nums2 = nums1 + C1 + 1
            nums3 = nums2 + 1
            nums4 = nums1 + 1
            numm1 = (C1 + 1) * (pxm - 1) + pym
            numm2 = numm1 + C1 + 1
            numm3 = numm2 + 1
            numm4 = numm1 + 1
            nume1 = (C1 + 1) * (pxe - 1) + pye
            nume2 = nume1 + C1 + 1
            nume3 = nume2 + 1
            nume4 = nume1 + 1

            Mesh_ps[:4, :] = [[X_col[pxs], Y_row[pys]], [X_col[pxs + 1], Y_row[pys]],
                              [X_col[pxs + 1], Y_row[pys + 1]], [X_col[pxs], Y_row[pys + 1]]]
            Mesh_pm[:4, :] = [[X_col[pxm], Y_row[pym]], [X_col[pxm + 1], Y_row[pym]],
                              [X_col[pxm + 1], Y_row[pym + 1]], [X_col[pxm], Y_row[pym + 1]]]
            Mesh_pe[:4, :] = [[X_col[pxe], Y_row[pye]], [X_col[pxe + 1], Y_row[pye]],
                              [X_col[pxe + 1], Y_row[pye + 1]], [X_col[pxe], Y_row[pye + 1]]]

            coeff_mesh_ps = mesh_grid_align(Mesh_ps, lps)
            coeff_mesh_pm = mesh_grid_align(Mesh_pm, lpm)
            coeff_mesh_pe = mesh_grid_align(Mesh_pe, lpe)

            sp_ii[24 * k: 24 * (k + 1)] = np.concatenate([[(2 * k - 1)] * 12, [2 * k] * 12])
            sp_jj[24 * k: 24 * (k + 1)] = [
                2 * nums1 - 1, 2 * nums2 - 1, 2 * nums3 - 1, 2 * nums4 - 1,
                2 * numm1 - 1, 2 * numm2 - 1, 2 * numm3 - 1, 2 * numm4 - 1,
                2 * nume1 - 1, 2 * nume2 - 1, 2 * nume3 - 1, 2 * nume4 - 1,
                2 * nums1, 2 * nums2, 2 * nums3, 2 * nums4,
                2 * numm1, 2 * numm2, 2 * numm3, 2 * numm4,
                2 * nume1, 2 * nume2, 2 * nume3, 2 * nume4
            ]
            sp_ss[24 * k: 24 * (k + 1)] = np.concatenate([
                -nor_vec[0] * coeff_mesh_ps, 2 * nor_vec[0] * coeff_mesh_pm, -nor_vec[0] * coeff_mesh_pe,
                -nor_vec[1] * coeff_mesh_ps, 2 * nor_vec[1] * coeff_mesh_pm, -nor_vec[1] * coeff_mesh_pe])
            k += 1

    # Construct sparse matrices
    S_matrix = csr_matrix((sp_s, (sp_i, sp_j)), shape=(k, 2 * num_V))
    S_matrix_equidistance = csr_matrix((sp_ss, (sp_ii, sp_jj)), shape=(2 * k, 2 * num_V))
    
    return S_matrix, S_matrix_equidistance
