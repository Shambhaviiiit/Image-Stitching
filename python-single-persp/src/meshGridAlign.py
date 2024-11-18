import numpy as np

def mesh_grid_align(Mesh, f_pts):
    """
    Calculate the coefficients for mesh grid alignment.

    Parameters:
    Mesh (numpy array): 4x2 matrix representing the coordinates of the mesh (x, y).
                        Mesh format:
                        v1 ________ v2
                          |  .p    |
                          |        |
                        v4|________| v3
    f_pts (numpy array): 1x2 array representing the feature point (px, py).

    Returns:
    numpy array: Coefficients for mesh grid alignment (w1, w2, w3, w4).
    """
    coeff_Mesh = np.zeros(4)  # Initialize coefficients w1, w2, w3, w4

    # Calculate the area of the mesh: (v2x - v1x) * (v4y - v1y)
    area_Mesh = (Mesh[1, 0] - Mesh[0, 0]) * (Mesh[3, 1] - Mesh[0, 1])

    # Calculate coefficients
    coeff_Mesh[0] = (Mesh[2, 0] - f_pts[0]) * (Mesh[2, 1] - f_pts[1])  # w1 = (v3x - px) * (v3y - py)
    coeff_Mesh[1] = (f_pts[0] - Mesh[3, 0]) * (Mesh[3, 1] - f_pts[1])  # w2 = (px - v4x) * (v4y - py)
    coeff_Mesh[2] = (f_pts[0] - Mesh[0, 0]) * (f_pts[1] - Mesh[0, 1])  # w3 = (px - v1x) * (py - v1y)
    coeff_Mesh[3] = (Mesh[1, 0] - f_pts[0]) * (f_pts[1] - Mesh[1, 1])  # w4 = (v2x - px) * (py - v2y)

    # Normalize by the area
    coeff_Mesh = coeff_Mesh / area_Mesh

    return coeff_Mesh
