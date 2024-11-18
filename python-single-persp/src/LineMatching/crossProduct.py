def crossproduct(p1, p2, o=[0, 0]):
    cp = (p1[0] - o[0]) * (p2[1] - o[1]) - (p2[0] - o[0]) * (p1[1] - o[1])
    return cp
