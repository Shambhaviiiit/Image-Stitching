import numpy as np
from scipy.sparse import csr_matrix
from src.calcSlope import calc_slope
from src.meshGridAlign import mesh_grid_align

def energy_line_segment(img, lines, slope_lines, init_H, C1, C2):
    num_V = (C1 + 1) * (C2 + 1)  # number of control vertices
    
    X_col = np.linspace(1, img.shape[1], C2 + 1)  # column index of cells
    Y_row = np.linspace(1, img.shape[0], C1 + 1)  # row index of cells
    x_dis = X_col[1] - X_col[0]  # the width of scale-cell
    y_dis = Y_row[1] - Y_row[0]  # the height of scale-cell
    
    Mesh_ps = np.zeros((4, 2))
    Mesh_pe = np.zeros((4, 2))
    
    # Sparse matrix components
    row_sp = np.sum(lines[::2, -1] - 1)  # row index
    sp_i = np.zeros(16 * row_sp, dtype=int)  # row index
    sp_j = np.zeros(16 * row_sp, dtype=int)  # column index
    sp_s = np.zeros(16 * row_sp)  # value index
    
    k = 0
    
    # Rotated vertical line equidistant-preserving
    for i in range(0, len(lines), 2):
        num_s = int(lines[i, -1])  # number of sample points in this segment
        k_xy = calc_slope(init_H, slope_lines[i], [lines[i, 0], lines[i+1, 0]])  # line's slope after transformation
        
        if np.isinf(abs(k_xy)):
            nor_vec = np.array([1, 0])
        else:
            nor_vec = np.array([k_xy, -1])
        
        nor_vec = nor_vec / np.linalg.norm(nor_vec)  # normal vector of warped lines
        
        for j in range(num_s - 1):
            lps = [lines[i, j], lines[i + 1, j]]
            lpe = [lines[i, j + 1], lines[i + 1, j + 1]]
            
            pxs = min(np.where(np.abs(lps[0] - X_col) < x_dis)[0][0], C2)
            pys = min(np.where(np.abs(lps[1] - Y_row) < y_dis)[0][0], C1)
            pxe = min(np.where(np.abs(lpe[0] - X_col) < x_dis)[0][0], C2)
            pye = min(np.where(np.abs(lpe[1] - Y_row) < y_dis)[0][0], C1)
            
            nums1 = (C1 + 1) * (pxs - 1) + pys
            nums2 = nums1 + C1 + 1
            nums3 = nums2 + 1
            nums4 = nums1 + 1
            nume1 = (C1 + 1) * (pxe - 1) + pye
            nume2 = nume1 + C1 + 1
            nume3 = nume2 + 1
            nume4 = nume1 + 1
            
            Mesh_ps[0:4, :] = np.array([
                [X_col[pxs], Y_row[pys]],
                [X_col[pxs + 1], Y_row[pys]],
                [X_col[pxs + 1], Y_row[pys + 1]],
                [X_col[pxs], Y_row[pys + 1]]
            ])
            
            Mesh_pe[0:4, :] = np.array([
                [X_col[pxe], Y_row[pye]],
                [X_col[pxe + 1], Y_row[pye]],
                [X_col[pxe + 1], Y_row[pye + 1]],
                [X_col[pxe], Y_row[pye + 1]]
            ])
            
            coeff_mesh_ps = mesh_grid_align(Mesh_ps, lps)
            coeff_mesh_pe = mesh_grid_align(Mesh_pe, lpe)
            
            sp_i[16 * k:16 * (k + 1)] = np.full(16, k)
            sp_j[16 * k:16 * (k + 1)] = np.array([
                2 * nums1 - 1, 2 * nums2 - 1, 2 * nums3 - 1, 2 * nums4 - 1,
                2 * nume1 - 1, 2 * nume2 - 1, 2 * nume3 - 1, 2 * nume4 - 1,
                2 * nums1, 2 * nums2, 2 * nums3, 2 * nums4,
                2 * nume1, 2 * nume2, 2 * nume3, 2 * nume4
            ])
            sp_s[16 * k:16 * (k + 1)] = np.concatenate([
                -nor_vec[0] * coeff_mesh_ps,
                nor_vec[0] * coeff_mesh_pe,
                -nor_vec[1] * coeff_mesh_ps,
                nor_vec[1] * coeff_mesh_pe
            ])
            
            k += 1
    
    # Construct the sparse matrix using scipy
    sparse_line = csr_matrix((sp_s, (sp_i, sp_j)), shape=(row_sp, 2 * num_V))
    
    return sparse_line
