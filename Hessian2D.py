import cv2
import numpy as np
from scipy.ndimage import convolve
def Hessian2D(I, Sigma):
    ndgrid = np.arange(-round(3*Sigma), round(3*Sigma)+1)
    X, Y = np.meshgrid(ndgrid, ndgrid)

    DGaussxx = 1/(2*np.pi*(Sigma**4))*(np.square(X)/Sigma**2-1)*np.exp(-(np.square(X)+np.square(Y))/(2*Sigma**2))
    DGaussxy = 1/(2*np.pi*(Sigma**6))*(X*Y)*np.exp(-(np.square(X)+np.square(Y))/(2*Sigma**2))
    DGaussyy = DGaussxx.T

    Dxx = convolve(I, DGaussxx, mode = 'constant')
    Dxy = convolve(I, DGaussxy, mode = 'constant')
    Dyy = convolve(I, DGaussyy, mode = 'constant')

    return Dxx, Dxy, Dyy