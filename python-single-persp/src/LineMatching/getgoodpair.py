import numpy as np

def getgoodpair(plines1, lines2, dist):
    len1 = len(plines1)
    len2 = len(lines2)
    
    print(f"getgoodpair input sizes: plines1={len1}, lines2={len2}")
    
    ind1 = []
    ind2 = []
    
    for i in range(len1):
        for j in range(len2):
            # Debug the first few iterations
            if i < 2 and j < 2:
                print(f"\nChecking line pair i={i}, j={j}")
                print(f"plines1[i]: {plines1[i]}")
                print(f"lines2[j]: {lines2[j]}")
                result = isclose(plines1[i], lines2[j], dist)
                print(f"isclose result: {result}")
            
            if isclose(plines1[i], lines2[j], dist):
                ind1.append(plines1[i]['ind'])
                ind2.append(j)
    
    print(f"getgoodpair found matches: {len(ind1)}")
    return ind1, ind2


def isclose(line1, line2, dist):
    print("\nisclose check:")
    d1 = disp2line(line1['point1'], line2)
    d2 = disp2line(line1['point2'], line2)
    d3 = disp2line(line2['point1'], line1)
    d4 = disp2line(line2['point2'], line1)
    
    midpoint_dist = np.linalg.norm((np.array(line1['point1']) + np.array(line1['point2'])) / 2 - 
                    (np.array(line2['point1']) + np.array(line2['point2'])) / 2)
    
    avg_length = (np.linalg.norm(np.array(line1['point1']) - np.array(line1['point2'])) + 
                 np.linalg.norm(np.array(line2['point1']) - np.array(line2['point2']))) / 2
    
    print(f"Distances to lines: {d1:.2f}, {d2:.2f}, {d3:.2f}, {d4:.2f}")
    print(f"Midpoint distance: {midpoint_dist:.2f}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Distance threshold: {dist}")
    
    if (d1 > dist or d2 > dist or d3 > dist or d4 > dist or 
        midpoint_dist > avg_length):
        return False
    return True


def disp2line(point, line):
    print("disp2line function")
    k = line['k']
    b = line['b']
    print(k, b)
    if k != float('inf'):
        dis = abs(k * point[0] - point[1] + b) / np.sqrt(k * k + 1)
    else:
        dis = abs(point[0] - line['point1'][0])
    
    print(dis)
    return dis
