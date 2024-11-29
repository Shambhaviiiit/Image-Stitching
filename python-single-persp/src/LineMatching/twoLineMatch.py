import numpy as np
import cv2
from pylsd2 import lsd
from skimage import io
from src.refineLine import refine_line
from src.LineMatching.getHpoints1L import get_Hpoints1L
from src.LineMatching.projline import projline, projpoint
from src.LineMatching.getgoodpair import getgoodpair
from src.linesDelete import linesDelete
from src.LineMatching.paras import paras
from src.LineMatching.addPointsNearby import addpointsnearby
from src.LineMatching.distline import distline

def two_line_match(img1, img2, pts1, pts2, parameters):
    # img1 = io.imread(imgpath1)
    # img2 = io.imread(imgpath2)
    print("2 LINE MATCH")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    lines_1 = lsd(img1)
    print(lines_1.shape)
    lines_1 = refine_line(lines_1, img1)
    print(lines_1)
    
    lines_2 = lsd(img2)
    lines_2 = refine_line(lines_2, img2)
    
    # Length of line segments
    len_lines1 = np.sqrt((lines_1[:, 2] - lines_1[:, 0])**2 + (lines_1[:, 3] - lines_1[:, 1])**2)
    len_lines2 = np.sqrt((lines_2[:, 2] - lines_2[:, 0])**2 + (lines_2[:, 3] - lines_2[:, 1])**2)
    
    # Apply length threshold
    len_threshold = parameters['line_threshold']
    
    lines1 = lines_1[len_lines1 >= len_threshold]
    lines2 = lines_2[len_lines2 >= len_threshold]
    
    # Reformat lines
    lines1 = np.column_stack([lines1[:, 0], lines1[:, 2], lines1[:, 1], lines1[:, 3]])
    lines2 = np.column_stack([lines2[:, 0], lines2[:, 2], lines2[:, 1], lines2[:, 3]])
    
    # Prepare points 
    lines1, pointlist1 = paras(img1, lines1)
    lines2, pointlist2 = paras(img2, lines2)
    # print("POINTLIST")
    # print(pointlist1)
    print("After paras")
    print(lines1.shape)
    print(pointlist1.shape)
    
    len1 = len(lines1)
    len2 = len(lines2)
    sublinds1 = list(range(len(lines1)))
    sublinds2 = list(range(len(lines2)))
    
    # Add points nearby 
    lines1 = addpointsnearby(lines1, pointlist1, sublinds1, pts1)
    lines2 = addpointsnearby(lines2, pointlist2, sublinds2, pts2)
    
    print("After points nearby")
    print(lines1.shape)
    # Initialize similarity matrices
    simL = np.zeros((len1, len2))
    simR = np.zeros((len1, len2))
    
    for i in range(len1):
        for j in range(len2):
            simL[i, j], simR[i, j] = distline(lines1[i], lines2[j])
    
    k = []
    
    # Find matching lines based on similarity threshold
    for i in range(len1):
        for j in range(len2):
            if simL[i, j] > 0.95 and simL[i, j] == np.max(simL[i, :]) and simL[i, j] == np.max(simL[:, j]):
                k.append([i, j])
                break
    
    simside1 = np.ones(len(k))
    
    for i in range(len1):
        for j in range(len2):
            if simR[i, j] > 0.95 and simR[i, j] == np.max(simR[i, :]) and simR[i, j] == np.max(simR[:, j]):
                k.append([i, j])
                break
    
    simside1 = np.concatenate([simside1, 2 * np.ones(len(k))])
    
    len_k = len(k)
    votecan = np.zeros((len2, len1))
    
    # Matching lines and calculating homographies
    for i in range(len_k):
        p1, p2 = get_Hpoints1L(lines1[k[i][0]], lines2[k[i][1]], simside1[i])
        if p1:
            F1, _, _ = cv2.findHomography(p1, p2, method=0)
            plines = projline(F1, lines1)
            ind11, ind12 = getgoodpair(plines, lines2, 3)
            plines = projline(np.linalg.inv(F1), lines2)
            ind22, ind21 = getgoodpair(plines, lines1, 3)
            
            if not ind11 or not ind22:
                continue
            
            indfinal = np.intersect1d(ind11, ind22)
            
            if indfinal.size > 0:
                ind1 = indfinal[0]
                ind2 = indfinal[1]
            else:
                ind1 = ind2 = []
            
            # Voting
            if simside1[i] == 1:
                v = simL[k[i][0], k[i][1]]
            else:
                v = simR[k[i][0], k[i][1]]
            
            votecan[ind2 + (ind1 - 1) * len2] += v
    
    num, ind = np.sort(votecan, axis=None)[::-1], np.argsort(votecan, axis=None)[::-1]
    num2, ind2 = np.sort(votecan.T, axis=None)[::-1], np.argsort(votecan.T, axis=None)[::-1]
    
    k = []
    for i in range(len(ind)):
        if ind[i] == ind2[ind[i]] and num[i] > 0.9 and num2[ind[i]] > 0.9:
            k.append(i)
    
    # Refine line matches
    linestruct1 = lines1[k]
    linestruct2 = lines2[ind[k]]
    linematch1 = np.zeros((len(linestruct1), 4))
    linematch2 = np.zeros_like(linematch1)
    
    for i in range(len(linestruct1)):
        linematch1[i, :2] = linestruct1[i][:2]
        linematch1[i, 2:] = linestruct1[i][2:]
        linematch2[i, :2] = linestruct2[i][:2]
        linematch2[i, 2:] = linestruct2[i][2:]
    
    linematch1 = refine_line(linematch1, img1)
    linematch2 = refine_line(linematch2, img2)
    
    linematch1, linematch2 = linesDelete(linematch1, linematch2, pts1, pts2)
    
    return linematch1, linematch2
