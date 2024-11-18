import numpy as np
from scipy.sparse import csr_matrix
from src.meshGridAlign import mesh_grid_align

def energy_line_align(img, C1, C2, line1, line2):
    M, N, _ = img.shape

    # Line's function: ax + by + c = 0
    abc_line2 = np.vstack([
        line2[:, 3] - line2[:, 1],
        line2[:, 0] - line2[:, 2],
        line2[:, 2] * line2[:, 1] - line2[:, 0] * line2[:, 3]
    ]).T

    # Generate mesh grid for spatial varying warp
    X_col = np.linspace(1, N, C2 + 1)  # column index of cells
    Y_row = np.linspace(1, M, C1 + 1)  # row index of cells
    num_V = (C1 + 1) * (C2 + 1)  # number of control vertices
    x_dis = X_col[1] - X_col[0]  # the width of scale-cell
    y_dis = Y_row[1] - Y_row[0]  # the height of scale-cell

    Mesh_p1 = np.zeros((4, 2))
    Mesh_p2 = np.zeros((4, 2))
    num_line = line1.shape[0]  # number of lines
    sp_i = np.ones(16 * num_line, dtype=int)  # row index
    sp_j = np.ones(16 * num_line, dtype=int)  # column index
    sp_s = np.zeros(16 * num_line)  # value index
    c_match = np.zeros(2 * num_line)  # match values
    
    # Optimize ||sparse_A*V_star||^2
    k = 0
    for i in range(num_line):
        a, b, c = abc_line2[i]
        d = np.sqrt(a ** 2 + b ** 2)

        # Find the line segments lying in which cell?
        px1 = np.min(np.where((line1[i, 0] - X_col) < x_dis & (line1[i, 0] - X_col) >= 0)[0], initial=C2)
        py1 = np.min(np.where((line1[i, 1] - Y_row) < y_dis & (line1[i, 1] - Y_row) >= 0)[0], initial=C1)
        px2 = np.min(np.where((line1[i, 2] - X_col) < x_dis & (line1[i, 2] - X_col) >= 0)[0], initial=C2)
        py2 = np.min(np.where((line1[i, 3] - Y_row) < y_dis & (line1[i, 3] - Y_row) >= 0)[0], initial=C1)

        # The cell containing line segments
        Mesh_p1[0] = [X_col[px1], Y_row[py1]]  # v1
        Mesh_p1[1] = [X_col[px1 + 1], Y_row[py1]]  # v2
        Mesh_p1[2] = [X_col[px1 + 1], Y_row[py1 + 1]]  # v3
        Mesh_p1[3] = [X_col[px1], Y_row[py1 + 1]]  # v4

        Mesh_p2[0] = [X_col[px2], Y_row[py2]]  # v1
        Mesh_p2[1] = [X_col[px2 + 1], Y_row[py2]]  # v2
        Mesh_p2[2] = [X_col[px2 + 1], Y_row[py2 + 1]]  # v3
        Mesh_p2[3] = [X_col[px2], Y_row[py2 + 1]]  # v4

        coeff_mesh_p1 = mesh_grid_align(Mesh_p1, line1[i, :2])
        coeff_mesh_p2 = mesh_grid_align(Mesh_p2, line1[i, 2:])

        num1 = (C1 + 1) * (px1 - 1) + py1
        num2 = num1 + (C1 + 1)
        num3 = num2 + 1
        num4 = num1 + 1

        num11 = (C1 + 1) * (px2 - 1) + py2
        num22 = num11 + (C1 + 1)
        num33 = num22 + 1
        num44 = num11 + 1

        sp_i[16 * k:16 * (k + 1)] = np.array([(2 * k - 1)] * 8 + [2 * k] * 8)
        sp_j[16 * k:16 * (k + 1)] = np.array([
            2 * num1 - 1, 2 * num2 - 1, 2 * num3 - 1, 2 * num4 - 1,
            2 * num1, 2 * num2, 2 * num3, 2 * num4,
            2 * num11 - 1, 2 * num22 - 1, 2 * num33 - 1, 2 * num44 - 1,
            2 * num11, 2 * num22, 2 * num33, 2 * num44
        ])
        sp_s[16 * k:16 * (k + 1)] = np.array([
            a / d * coeff_mesh_p1, b / d * coeff_mesh_p1, a / d * coeff_mesh_p2, b / d * coeff_mesh_p2
        ]).flatten()

        c_match[2 * k:2 * (k + 1)] = np.array([-c / d, -c / d])

        k += 1

    sparse_lineal = csr_matrix((sp_s, (sp_i - 1, sp_j - 1)), shape=(2 * num_line, 2 * num_V))

    return sparse_lineal, c_match
