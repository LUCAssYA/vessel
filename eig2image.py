import numpy as np
def eig2image(Dxx, Dxy, Dyy):
    tmp = np.sqrt(np.square((Dxx-Dyy))+4*np.square(Dxy))
    v1x, v1y = 2*Dxy, Dyy-Dxx+tmp

    mag = np.sqrt(np.square(v1x)+np.square(v1y))
    i = (mag != 0)
    v1x[i] = v1x[i]/mag[i]
    v1y[i] = v1y[i]/mag[i]

    v2x = -v1y
    v2y = v1x.copy()

    mu2 = 0.5*(Dxx+Dyy+tmp)
    mu1 = 0.5*(Dxx+Dyy-tmp)

    check = abs(mu1)>abs(mu2)

    Lambda1 = mu1.copy()
    Lambda1[check] = mu2[check]
    Lambda2 = mu2.copy()
    Lambda2[check] = mu1[check]

    Ix = v2x.copy()
    Ix[check] = v2x[check]
    Iy = v2y.copy()
    Iy[check] = v2y[check]

    return Lambda1, Lambda2, Ix, Iy, mu1, mu2