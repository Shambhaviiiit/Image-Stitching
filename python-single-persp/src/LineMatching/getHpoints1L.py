import numpy as np

def get_Hpoints1L(line1, line2, side):
    minnum = 10
    p1 = []
    p2 = []
    
    if side == 1:
        # Assuming line1.pleft and line2.pleft are numpy arrays
        C, ind1, ind2 = np.intersect1d(line1['pleft'][:, 2], line2['pleft'][:, 2], return_indices=True)
        n = len(C)
        
        if n >= minnum:
            p1 = line1['pleft'][ind1, :2]
            p2 = line2['pleft'][ind2, :2]
    
    elif side == 2:
        # Assuming line1.pright and line2.pright are numpy arrays
        C, ind1, ind2 = np.intersect1d(line1['pright'][:, 2], line2['pright'][:, 2], return_indices=True)
        n = len(C)
        
        if n >= minnum:
            p1 = line1['pright'][ind1, :2]
            p2 = line2['pright'][ind2, :2]
    
    return p1, p2
