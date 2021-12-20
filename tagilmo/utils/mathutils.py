import math

def degree2rad(angle):
    return angle * math.pi / 180

def normAngle(angle):
    while (angle < -math.pi): angle += 2 * math.pi
    while (angle > math.pi): angle -= 2 * math.pi
    return angle

def toRadAndNorm(angle):
    return normAngle(degree2rad(angle))

def int_coord(x):
    '''
    A proper way to turn float coordinates into integer coordinates of block in Minecraft
    Add 0.5 to int coord to get real coordinate of block center
    '''
    return math.floor(x)

def int_coords(xs):
    return list(map(int_coord, xs))

def dist_vec(v1, v2):
    d = 0
    for c1, c2 in zip(v1, v2):
        d += (c1 - c2) * (c1 - c2)
    return math.sqrt(d)

