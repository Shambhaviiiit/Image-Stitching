import numpy as np

def get_Hpoints1L(line1, line2, side):
    minnum = 10
    p1 = []
    p2 = []
    # print("pleft")
    # print(line1['pleft'])
    
    if side == 1:
        # Assuming line1.pleft and line2.pleft are numpy arrays
        # Check if pleft exists and is not empty
        if 'pleft' not in line1 or 'pleft' not in line2 or \
           not line1['pleft'] or not line2['pleft']:
            return [], []
        C, ind1, ind2 = np.intersect1d(np.array(line1['pleft'])[:, 2], np.array(line2['pleft'])[:, 2], return_indices=True)
        n = len(C)
        
        if n >= minnum:
            p1 = np.array(line1['pleft'])[ind1, :2]
            p2 = np.array(line2['pleft'])[ind2, :2]
    
    elif side == 2:
        # Assuming line1.pright and line2.pright are numpy arrays
        # Check if pright exists and is not empty
        if 'pright' not in line1 or 'pright' not in line2 or \
           not line1['pright'] or not line2['pright']:
            return [], []
        
        C, ind1, ind2 = np.intersect1d(np.array(line1['pright'])[:, 2], np.array(line2['pright'])[:, 2], return_indices=True)
        n = len(C)
        
        if n >= minnum:
            p1 = np.array(line1['pright'])[ind1, :2]
            p2 = np.array(line2['pright'])[ind2, :2]
    
    return p1, p2