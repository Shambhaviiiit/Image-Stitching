import numpy as np
from src.LineMatching.sameside import sameside
from src.LineMatching.getPoints import getpoints
from src.LineMatching.crossProduct import crossproduct

def addpointsnearby(lines, pointlist, sublinds, charap):
    print("ADD POINTS NEARBY")
    print(charap)
    canlines = [lines[i] for i in sublinds]
    for i in range(len(sublinds)):
        # Get intersections
        canlines[i]['intsect1'], canlines[i]['intsect2'] = get2imps(sublinds[i], lines, pointlist)
        # Add character approximation rectangle
        canlines[i]['pleft'], canlines[i]['pright'] = addcharapsrect(lines[sublinds[i]], charap)

    canlines = np.array(canlines)
    return canlines

def get2imps(lind, lines, pointlist):
    subline = lines[lind]
    lnum = len(subline)
    p1 = np.zeros((lnum, 2))
    p2 = np.zeros((lnum, 2))
    
    print("GET 2 IMPS")
    print(lind)
    # print(lines.shape)
    # print(pointlist.shape)

    pinds = getpoints(lind, pointlist)
    n = len(pinds)
    print("AFTER GET POINTS")
    print(n)
    points = np.zeros((n, 2))
    dist1 = np.zeros(n)
    dist2 = np.zeros(n)
    
    for j in range(n):
        p = np.array(pointlist[pinds[j]]['point'])
        points[j, :] = p
        dist1[j] = np.linalg.norm(p - np.array(subline['point1']))
        dist2[j] = np.linalg.norm(p - np.array(subline['point2']))
    
    d1 = np.min(dist1) if len(dist1) > 0 else np.inf
    id1 = np.argmin(dist1) if len(dist1) > 0 else -1
    
    d2 = np.min(dist2) if len(dist2) > 0 else np.inf
    id2 = np.argmin(dist2) if len(dist2) > 0 else -1

    # linelen = np.linalg.norm(subline['point1'] - subline['point2']) / 10
    linelen = np.linalg.norm(np.array(subline['point1']) - np.array(subline['point2'])) / 10

    
    if d1 > linelen and d2 <= linelen:
        p1 = subline['point1']
        p2 = points[id2, :]
    elif d2 > linelen and d1 <= linelen:
        p2 = subline['point2']
        p1 = points[id1, :]
    elif d1 <= linelen and d2 <= linelen:
        p1 = points[id1, :]
        p2 = points[id2, :]
    else:
        p1 = subline['point1']
        p2 = subline['point2']

    mid = (np.array(subline['point1']) + np.array(subline['point2']) )/ 2
    endp = mid + subline['gradient']
    cp = crossproduct(p1, endp, mid)
    
    if cp < 0:
        p1, p2 = p2, p1

    return p1, p2

def addcharapsrect(line, charap):
    d2line = 2
    d2midline = 0.5
    pleft = []
    pright = []
    
    mid = (np.array(line['point1']) + np.array(line['point2'])) / 2
    pg = mid + line['gradient']
    linelen = np.linalg.norm(np.array(line['point1']) - np.array(line['point2']))
    
    # Midline calculation
    if line['k'] != 0:
        midline = {'k': -1 / line['k']}
        if line['k'] == float('inf'):
            midline['b'] = line['point1'][0]
        else:
            midline['b'] = (line['k'] * mid[1] + mid[0]) / line['k']
    else:
        midline = {'k': float('inf'), 'b': float('inf')}
    
    for i in range(len(charap)):
        p = charap[i]
        if disp2line(p, line) < d2line * linelen and disp2line(p, midline) < d2midline * linelen:
            if sameside(line, p, pg):
                pright.append([p, i])
            else:
                pleft.append([p, i])
    
    return pleft, pright

def disp2line(point, line):
    k = line['k']
    b = line['b']

    print("DISP2LINE")
    print(point)
    print(line)
    
    if k != float('inf'):
        return abs(k * point[0] - point[1] + b) / np.sqrt(k ** 2 + 1)
    else:
        return abs(point[0] - line['point1'][0])