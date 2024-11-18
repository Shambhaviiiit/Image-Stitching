def getpoints(linenum, pointlist, except_line=0):
    inds = []
    for i in range(len(pointlist)):
        if (linenum in pointlist[i]['lines']) and (except_line not in pointlist[i]['lines']):
            inds.append(i)
    return inds
