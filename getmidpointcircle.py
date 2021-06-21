import numpy as np

def getmidpointcircle(x0, y0, radius):
    octant_size = int(np.floor((np.sqrt(2)*(radius-1)+4)/2))
    n_points = int(8*octant_size)

    xc, yc = np.empty((n_points, 1)), np.empty((n_points, 1))
    xc[:], yc[:] = np.nan, np.nan

    x = 0
    y = radius
    f = 1-radius
    dx = 1
    dy = -2*radius

    #1st octant
    xc[0] = x0+x
    yc[0] = y0+y

    #2nd octant
    xc[8*octant_size-1] = x0-x
    yc[8*octant_size-1] = y0+y

    #3rd octant
    xc[4*octant_size-1] = x0+x
    yc[4*octant_size-1] = y0-y

    #4th octant
    xc[4*octant_size] = x0-x
    yc[4*octant_size] = y0-y

    #5th octant
    xc[2*octant_size-1] = x0+y
    yc[2*octant_size-1] = y0+x

    #6th octant
    xc[6*octant_size] = x0-y
    yc[6*octant_size] = y0+x

    #7th octant
    xc[2*octant_size] = x0+y
    yc[2*octant_size] = y0-x

    #8th octant
    xc[6*octant_size-1] = x0-y
    yc[6*octant_size-1] = y0-x

    for i in range(1, n_points//8):
        if f > 0:
            y -= 1
            dy += 2
            f += dy

        x +=1
        dx += 2
        f += dx

        #1st octant
        xc[i] = x0+x
        yc[i] = y0+y

        #2nd octant

        xc[8*octant_size-1-i] = x0-x
        yc[8*octant_size-1-i] = y0+y

        #3rd octant

        xc[4*octant_size-1-i] = x0+x
        yc[4*octant_size-1-i] = y0-y

        #4th octant

        xc[4*octant_size+i] = x0-x
        yc[4*octant_size+i] = y0-y

        #5th octant

        xc[2*octant_size-1-i] = x0+y
        yc[2*octant_size-1-i] = y0+x

        #6th octant

        xc[6*octant_size+i] = x0-y
        yc[6*octant_size+i] = y0+x

        #7th octant

        xc[2*octant_size+i] = x0+y
        yc[2*octant_size+i] = y0-x

        #8th octant

        xc[6*octant_size-1-i] = x0-y
        yc[6*octant_size-1-i] = y0-x

    xc2 = np.zeros((0, 1))
    yc2 = np.zeros((0, 1))
    for i in range(n_points-1):
        if not (xc[i+1] == xc[i] and yc[i+1] == yc[i]):
            xc2 = np.append(xc2, np.array([xc[i]]), axis = 0)
            yc2 = np.append(yc2, np.array([yc[i]]), axis = 0)
    if not (xc[n_points-1]==xc[0] and yc[n_points-1]==yc[0]):
        xc2 = np.append(xc2, xc[n_points], axis = 0)
        yc2 = np.append(yc2, yc[n_points], axis = 0)




    return xc2, yc2