import numpy as np
import cv2
from matplotlib import pyplot as plt

def lines_detect(imgpath, img, C1, C2):
    # Load image and convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply LSD (Line Segment Detector)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _ = lsd.detect(gray_img)
    
    M, N, _ = img.shape
    x_dis = (N - 1) / C2
    y_dis = (M - 1) / C1
    
    len_threshold = np.sqrt(x_dis**2 + y_dis**2)
    
    # Filter lines based on length threshold
    lines_ = []
    for line in lines[0]:
        x1, y1, x2, y2, _, _ = line
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length >= len_threshold:
            lines_.append([x1, y1, x2, y2, line_length])
    
    # Convert to numpy array for easier manipulation
    lines_ = np.array(lines_)
    
    # Sample points on lines
    num_lines = np.maximum(3, 2 * np.round(lines_[:, 4] / np.minimum(x_dis, y_dis)).astype(int))
    
    sample_lines = np.zeros((2 * len(lines_), max(num_lines) + 1))
    slope_lines = np.zeros(2 * len(lines_))
    
    for k, line in enumerate(lines_):
        x1, y1, x2, y2, _ = line
        
        if x1 != x2:
            slope = (y2 - y1) / (x2 - x1)
            xseq = np.linspace(x1, x2, num_lines[k])
            yseq = (xseq - x1) * slope + y1
            r_seq = (xseq >= 1) & (xseq <= N) & (yseq >= 1) & (yseq <= M)
            
            sample_lines[2*k:2*k+2, :np.sum(r_seq)] = np.vstack([xseq[r_seq], yseq[r_seq]])
            sample_lines[2*k:2*k+2, -1] = np.sum(r_seq)
            slope_lines[2*k:2*k+2] = slope
        
        elif x1 == x2:
            xseq = np.ones(num_lines[k]) * x1
            yseq = np.linspace(y1, y2, num_lines[k])
            r_seq = (xseq >= 1) & (xseq <= N) & (yseq >= 1) & (yseq <= M)
            
            sample_lines[2*k:2*k+2, :np.sum(r_seq)] = np.vstack([xseq[r_seq], yseq[r_seq]])
            sample_lines[2*k:2*k+2, -1] = np.sum(r_seq)
            slope_lines[2*k:2*k+2] = np.inf
    
    return sample_lines, slope_lines

# Example usage:
# imgpath = 'path_to_image'
# img = cv2.imread(imgpath)
# C1, C2 = 5, 5  # Example grid sizes
# sample_lines, slope_lines = linesDetect(imgpath, img, C1, C2)

# Plot the result
# plt.imshow(img)
# plt.plot(sample_lines[0, :], sample_lines[1, :], 'ro')
# plt.show()
