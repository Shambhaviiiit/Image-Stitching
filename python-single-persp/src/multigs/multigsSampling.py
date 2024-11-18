import numpy as np
import time
import warnings
from src.modelspecific.homographic_degen import homography_degen
from src.modelspecific.homographic_res import homography_res
from src.modelspecific.homographic_fit import homography_fit
from src.multigs import computeIntersection
import random

#---------------------------
# Model specific parameters.
#---------------------------
# Define globally as needed:
# fitfn, resfn, degenfn, psize, numpar

# compute intersection is implemented by GPT, original is in C

def multigs_sampling(lim, data, M, blksiz):
    """
    Perform multi-structure robust fitting with guided sampling.
    
    Parameters:
    lim (float): Maximum CPU seconds allowed.
    data (np.ndarray): Input data of shape (d, n).
    M (int): Maximum number of hypotheses to be generated.
    blksiz (int): Block size of Multi-GS.
    
    Returns:
    par (np.ndarray): Parameters of the putative models.
    res (np.ndarray): Residuals to the putative models.
    inx (np.ndarray): Indices of p-subsets.
    tim (np.ndarray): CPU time for generating each model.
    err (int): Error status (0 if no error, 1 otherwise).
    """
    
    global psize, numpar
    
    if M % blksiz != 0:
        raise ValueError("Bad block size!")
    

    numpar = 9
    psize = 4
    n = data.shape[1]
    par = np.zeros((numpar, M))
    res = np.zeros((n, M))
    inx = np.zeros((psize, M), dtype=int)
    tim = np.zeros(M)
    err = 0
    
    print(f'Multi-GS (RANSAC) sampling for {lim:.2f} seconds...')
    t0 = time.time()
    
    for m in range(M):
        print(m)
        degencnt = 0
        isdegen = True
        
        while isdegen and degencnt <= 10:
            degencnt += 1
            # if m < blksiz:
            #     pinx = np.random.choice(n, psize, replace=False)
            print(f"N VALUE: {n}")
            if m <= blksiz:
                if psize > n:
                    pinx = np.random.choice(n, psize, replace=True)  # Allow sampling with replacement
                else:
                    pinx = np.random.choice(n, psize, replace=False)

            else:
                pinx = weighted_sampling(n, psize, resinx, win)
                
            print("PINX")
            print(pinx)
            psub = data[:, pinx]
            
            isdegen = homography_degen(psub)
        
        print("out of while")
        if isdegen:
            warnings.warn("Cannot find a valid p-subset!")
            err = 1
            return par, res, inx, tim, err
        
        # Fit the model on the p-subset
        print("PSUB: ")
        print(psub.shape)
        st = homography_fit(psub)
        print("ST")
        # print(st[0])
        # print(st[1])
        
        # Compute residuals
        ds = homography_res(st[0], data)
        print("returned res")
        print("DS shapes")
        print(ds[0].shape)
        print(ds[1].shape)
        
        # Store results
        par[:, m] = st[0]
        res[:, m] = ds[0].flatten()
        print("res 1 done")
        inx[:, m] = pinx
        tim[m] = time.time() - t0
        
        if tim[m] >= lim:
            par = par[:, :m + 1]
            res = res[:, :m + 1]
            print("res 2 in if done")
            inx = inx[:, :m + 1]
            tim = tim[:m + 1]
            break
        
        if m >= blksiz and m % blksiz == 0:
            win = round(0.1 * m)
            resinx = np.argsort(res[:, :m + 1], axis=1)

        print("loop end")
    
    print(f'done ({tim[-1]:.2f}s)')
    return par, res, inx, tim, err

def weighted_sampling(n, psize, resinx, win):
    """
    Perform weighted sampling.
    
    Parameters:
    n (int): Size of data.
    psize (int): Size of p-subset.
    resinx (np.ndarray): Indices of sorted hypotheses for each datum.
    win (int): Intersection width.
    
    Returns:
    pinx (np.ndarray): Indices of the p-subset.
    """
    
    pinx = np.zeros(psize, dtype=int)
    seedinx = np.random.choice(n)
    pinx[0] = seedinx
    
    w = np.ones(n)
    # print(win.shape)
    for i in range(1, psize):
        print("ITERARION")
        print(i)
        # print(seedinx)
        selected_row = resinx[seedinx, :].reshape(resinx.shape[1], 1)

        # print("selected row:", selected_row.shape)
        # print("resinx shape:", resinx.T.shape)
        print("win value:", win)

        new_w = computeIntersection.compute_intersection(selected_row, resinx.T, win).T
        print(new_w.shape)
        new_w[seedinx] = 0  # Ensure sampling without replacement
        
        print("SHAPES")
        # print("w: ")
        # print(w.shape)
        # print("new_w: ")
        # print(new_w.shape)
        w = w.reshape(w.shape[0], 1)
        w *= new_w
        # print("done")
        othinx = np.random.choice(n)
        if np.sum(w) > 0:
            # p=normalize_weights(w)
            # print(p)
            othinx = random.choices(range(0, n), weights=w, k=1)[0]
        # else:
            # pinx[i] = np.random.choice(n)

        pinx[i] = othinx
        seedinx = othinx
    
    return pinx

# def compute_intersection(seedinx, resinx, win):
#     """
#     Compute intersection weights.
    
#     Parameters:
#     seedinx (np.ndarray): Seed index array.
#     resinx (np.ndarray): Indices of sorted hypotheses.
#     win (int): Intersection width.
    
#     Returns:
#     new_w (np.ndarray): Computed intersection weights.
#     """
#     new_w = np.zeros(resinx.shape[1])
#     for j in range(resinx.shape[0]):
#         new_w[resinx[j, :win]] += 1
#     return new_w

def normalize_weights(w):
    """ Normalize weights to make them a probability distribution. """
    return w / np.sum(w)
