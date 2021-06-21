import numpy as np
from Hessian2D import Hessian2D
from eig2image import eig2image
def FrangiFilter2D(I, **options):
    flag_calc_original_frangi = False
    flag_use_original_frangi = False

    if np.max(I) <= 1:
        I = I*255
    defaultoptions = {"FrangiScaleRange":[4, 6], "FrangiScaleRatio":1, "FrangiBetaOne":0.5,
                      "FrangiBetaTwo":15, "verbose":True, "BlackWhite":True}

    if options == {}:
        options = defaultoptions
    else:
        tags = defaultoptions.keys()
        for i in tags:
            if i not in options:
                options[i] = defaultoptions[i]

    sigmas = list(range(options["FrangiScaleRange"][0], options["FrangiScaleRange"][1]+1))
    beta = 2*(options["FrangiBetaOne"]**2)
    c = 2*(options["FrangiBetaTwo"]**2)

    ALLfiltered = np.zeros((I.shape[0], I.shape[1], len(sigmas)))
    Allangles = np.zeros_like(ALLfiltered)
    Allmu1 = np.zeros_like(ALLfiltered)
    Allmu2 = np.zeros_like(ALLfiltered)

    for i in range(len(sigmas)):
        if options['verbose']:
            print("Current Frangi Filter Sigma %d"%(sigmas[i]))

        Dxx, Dxy, Dyy = Hessian2D(I, sigmas[i])

        Dxx, Dxy, Dyy = (sigmas[i]**2)*Dxx, (sigmas[i]**2)*Dxy, (sigmas[i]**2)*Dyy

        Lambda1, Lambda2, Ix, Iy, mu1, mu2 = eig2image(Dxx, Dxy, Dyy)

        angles = np.arctan2(Iy, Ix)
        Lambda2[Lambda2 == 0] = np.spacing(1)

        Rb = np.square((Lambda1/Lambda2))

        if options['BlackWhite']:
            mu1[mu1<0] = 0
            mu2[mu2<0] = 0
        else:
            mu1[mu1>0] = 0
            mu2[mu2>0] = 0

        S2 = np.square(mu1)+np.square(mu2)

        Ifiltered = np.ones_like(I)-np.exp(-S2/c)
        ALLfiltered[:, :, i] = Ifiltered
        Allangles[:, :, i] = angles
        Allmu1[:, :, i] = mu1
        Allmu2[:, :, i] = mu2
        if flag_calc_original_frangi:
            S2_ori = np.square(Lambda1)+np.square(Lambda2)
            Ifiltered_ori = np.exp(-Rb/beta)*(np.ones_like(I)-np.exp(-S2_ori/c))
            if options['BlackWhite']:
                Ifiltered_ori[Lambda2<0] = 0
            else:
                Ifiltered_ori[Lambda2>0] = 0
            if flag_use_original_frangi:
                ALLfiltered[:, :, i] = Ifiltered_ori
    outDoH = Allmu1[:, :, 0] * Allmu2[:, :, 0]

    for i in range(1, len(sigmas)):
        outDoH = np.maximum(outDoH, Allmu1[:, :, i]*Allmu2[:, :, i])

    if len(sigmas) > 1:
        outIm = np.max(ALLfiltered, axis = 2)
        outIm = np.reshape(outIm, I.shape)

    return outIm, outDoH