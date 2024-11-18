import numpy as np
from scipy.sparse import coo_matrix
from src.meshGridAlign import mesh_grid_align

def energy_align(img, C1, C2, pts1, pts2):
    # For given target image, generate C1*C2 mesh grid for feature alignment
    scale = 1  # The scale operator for alignment, scale: 1-scale
    M, N, _ = img.shape
    
    # Generating mesh grid for spatial varying warp
    X_col = np.linspace(1, N, C2 + 1)  # Column index of cells
    Y_row = np.linspace(1, M, C1 + 1)  # Row index of cells
    num_V = (C1 + 1) * (C2 + 1)  # Number of control vertices
    x_dis = X_col[1] - X_col[0]  # Width of scale-cell
    y_dis = Y_row[1] - Y_row[0]  # Height of scale-cell
    max_length = sum((np.arange(1, scale + 1))**2)

    cell_sparse = [None] * scale
    psMatch = np.zeros(max_length * 2 * pts1.shape[1], dtype=np.float64)

    start = 0
    Mesh_p = np.zeros((4, 2))
    
    for s in range(1, scale + 1):
        num_spts = 2 * s**2 * pts1.shape[1]  # The number of sparse points
        sp_i = np.ones(4 * num_spts, dtype=int)  # Row index
        sp_j = np.ones(4 * num_spts, dtype=int)  # Column index
        sp_s = np.zeros(4 * num_spts, dtype=np.float64)  # Value index (ignored zero-element)
        pmatch = np.zeros(num_spts, dtype=np.float64)  # Feature matches in reference image

        k = 0
        for i in range(pts1.shape[1]):
            # Find the feature point lies in which cell?
            px = np.min(np.where((pts1[0, i] - X_col) < x_dis & (pts1[0, i] - X_col) >= 0)[0], initial=C2)
            py = np.min(np.where((pts1[1, i] - Y_row) < y_dis & (pts1[1, i] - Y_row) >= 0)[0], initial=C1)

            # Count all the scale-cells
            for xi in range(px - s + 1, px + 1):
                for yi in range(py - s + 1, py + 1):
                    if xi > 0 and yi > 0 and xi + s <= C2 + 1 and yi + s <= C1 + 1:  # If the scale-cell is well-defined
                        # The cell containing feature p
                        Mesh_p[0, :] = [X_col[xi - 1], Y_row[yi - 1]]  # v1
                        Mesh_p[1, :] = [X_col[xi + s - 1], Y_row[yi - 1]]  # v2
                        Mesh_p[2, :] = [X_col[xi + s - 1], Y_row[yi + s - 1]]  # v3
                        Mesh_p[3, :] = [X_col[xi - 1], Y_row[yi + s - 1]]  # v4
                        
                        coeff_mesh_p = mesh_grid_align(Mesh_p, pts1[:, i])

                        num1 = (C1 + 1) * (xi - 1) + yi
                        num2 = num1 + s * (C1 + 1)
                        num3 = num2 + s
                        num4 = num1 + s
                        
                        sp_i[8 * k:8 * k + 4] = [(2 * k - 1)] * 4 + [2 * k] * 4
                        sp_j[8 * k:8 * k + 4] = [
                            2 * num1 - 1, 2 * num2 - 1, 2 * num3 - 1, 2 * num4 - 1,
                            2 * num1, 2 * num2, 2 * num3, 2 * num4
                        ]
                        sp_s[8 * k:8 * k + 4] = [coeff_mesh_p, coeff_mesh_p]
                        pmatch[2 * k:2 * k + 2] = pts2[0:2, i]
                        
                        k += 1

        sp_s = np.sqrt(1.0 / s) * sp_s
        pmatch = np.sqrt(1.0 / s) * pmatch

        sparse_sa = coo_matrix((sp_s, (sp_i, sp_j)), shape=(num_spts, 2 * num_V))
        tmp_length = len(pmatch)
        psMatch[start:start + tmp_length] = pmatch
        start += tmp_length

        cell_sparse[s - 1] = sparse_sa  # The alignment term under scale s
    
    sparse_al = coo_matrix(([], ([], [])), shape=(0, 2 * num_V))
    for i in range(scale):
        sparse_al = coo_matrix(np.vstack([sparse_al.toarray(), cell_sparse[i].toarray()]))
    
    return sparse_al, psMatch
