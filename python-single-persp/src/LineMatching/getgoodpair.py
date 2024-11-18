import numpy as np

def getgoodpair(plines1, lines2, dist):
    len1 = len(plines1)
    len2 = len(lines2)
    
    ind1 = []
    ind2 = []
    
    for i in range(len1):
        for j in range(len2):
            if isclose(plines1[i], lines2[j], dist):
                ind1.append(plines1[i]['ind'])
                ind2.append(j)
    
    return ind1, ind2


def isclose(line1, line2, dist):
    if (disp2line(line1['point1'], line2) > dist or 
        disp2line(line1['point2'], line2) > dist or 
        disp2line(line2['point1'], line1) > dist or 
        disp2line(line2['point2'], line1) > dist or 
        np.linalg.norm((np.array(line1['point1']) + np.array(line1['point2'])) / 2 - 
                       (np.array(line2['point1']) + np.array(line2['point2'])) / 2) > 
        (np.linalg.norm(np.array(line1['point1']) - np.array(line1['point2'])) + 
         np.linalg.norm(np.array(line2['point1']) - np.array(line2['point2']))) / 2):
        return False
    else:
        return True


def disp2line(point, line):
    k = line['k']
    b = line['b']
    
    if k != float('inf'):
        dis = abs(k * point[0] - point[1] + b) / np.sqrt(k * k + 1)
    else:
        dis = abs(point[0] - line['point1'][0])
    
    return dis
