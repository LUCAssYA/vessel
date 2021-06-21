import cv2
import numpy as np
from FrangiFilter2D import FrangiFilter2D
from getmidpointcircle import getmidpointcircle
from find_connected_components import find_connected_components
from nonmax_suppression import nonmax_suppresion

def main(path):
    img = cv2.imread(path)
    radius = [11, 7]
    thresh_specular = 255
    feature_points = detector_VBCT(img, radius, thresh_specular)

def VBCT(Ivessel, radius, DoH):
    distance_closet_feature_point = 10
    num_smallest_component = 3
    margin = np.array(radius)*2
    thresh_black = 0.01
    thresh_mu = 0.05
    k = 1
    mark_candidate = np.logical_and((DoH>thresh_mu), (Ivessel>thresh_black))

    if np.max(Ivessel)>1:
        Ivessel = Ivessel/255.0

    imsz = np.array(Ivessel.shape)
    imcenter = np.floor(imsz/2)

    circley_temp, circlex_temp = getmidpointcircle(imcenter[0], imcenter[1], radius[0])
    circle_templatex1 = circlex_temp - imcenter[1]
    circle_templatey1 = circley_temp - imcenter[0]
    circley_temp, circlex_temp = getmidpointcircle(imcenter[0], imcenter[1], radius[1])
    circle_templatex2 = circlex_temp-imcenter[1]
    circle_templatey2 = circley_temp - imcenter[0]

    branch_points = np.array([])
    peak_points = np.array([])

    for c in range(margin[0]-1, imsz[1]-margin[0]):
        for r in range(margin[0]-1, imsz[0]-margin[0]):
            if mark_candidate[r, c] == 0:
                continue

            circlex1 = circle_templatex1+(c+1)
            circley1 = circle_templatey1+(r+1)
            num_pixel1 = len(circlex1)
            distribution1 = np.zeros((num_pixel1, 1))

            for i in range(num_pixel1):
                px = int(circlex1[i])
                py = int(circley1[i])
                distribution1[i] = Ivessel[py-1, px-1]
            isbranch1, peaks1 = circle_test_vesselness(distribution1, Ivessel[r,c])
            circlex2 = circle_templatex2+(c+1)
            circley2 = circle_templatey2+(r+1)

            num_pixel2 = len(circlex2)
            distribution2 = np.zeros((num_pixel2, 1))
            for i in range(num_pixel2):
                px = int(circlex2[i])
                py = int(circley2[i])
                distribution2[i] = Ivessel[py-1, px-1]
            isbranch2, peaks2 = circle_test_vesselness(distribution2, Ivessel[r, c])
            bp = np.array([])
            if isbranch1 or isbranch2:
                bp = np.append(bp, c+1)
                bp = np.append(bp, r+1)
                len_peaks_allowed = 4

                if isbranch1:
                    len_peaks_allowed = min(len_peaks_allowed, len(peaks1))
                    idx = [i-1 for i in peaks1]
                    pp1 = circlex1[idx[0:len_peaks_allowed]].squeeze()
                    pp2 = circley1[idx[0:len_peaks_allowed]].squeeze()
                else:
                    len_peaks_allowed = min(len_peaks_allowed, len(peaks2))
                    idx = [i-1 for i in peaks2]
                    pp1 = circlex2[idx[0:len_peaks_allowed]].squeeze()
                    pp2 = circley2[idx[0:len_peaks_allowed]].squeeze()
                if len_peaks_allowed < 4:
                    for i in range(len(pp1)-2):
                        pp1 = np.append(pp1, 0)
                        pp2 = np.append(pp2,0)
                k += 1

                branch_points = np.vstack((branch_points, bp)) if branch_points.size != 0 else bp
                peak_points = np.vstack((peak_points, pp1)) if peak_points.size != 0 else pp1
                peak_points = np.vstack((peak_points, pp2))
    connected_components, tag_which_component, connected_peaks = find_connected_components(branch_points, peak_points, imsz)
    feature_points, ridges = nonmax_suppresion(Ivessel, connected_components, connected_peaks, distance_closet_feature_point, num_smallest_component)

    return feature_points, ridges






    return feature_points, ridges
def detector_VBCT(I, radius, thresh_specular):
    img_orig = I.copy()
    flag_show_result = True
    I = I/255.0

    if len(I.shape) > 2 and I.shape[2] > 1:
        I = I[:, :, 1]

    I = np.array(I, dtype = np.float)/np.max(I)
    option = {"FrangiScaleRange": [3, 5], "FrangiScaleRatio":1, "FrangiBetaOne": 0.5,
              "FrangiBetaTwo":15, "verbose": True, "BlackWhite": True}
    vessel, DoH = FrangiFilter2D(I, **option)

    feature_points, ridges = VBCT(vessel, radius, DoH)

    if flag_show_result:
        for x, y in feature_points[:, :2]:
            img_orig = cv2.circle(img_orig, (int(x), int(y)), 3, (0, 255, 0), 1)

        cv2.imshow("result", img_orig)
        cv2.waitKey(0)
        cv2.destroyWindow()


    return feature_points, ridges, vessel
def circle_test_vesselness(distribution, val_center):
    isbranch = False

    ridge = []

    num = len(distribution)

    k = 1

    thresh_similar = 0.03
    thresh_black = 0.005
    thresh_bright = 0.02
    thresh_ratio_peaks = 1/10
    thresh_monotonic = 0.0001
    thresh_peak_height = 0.005
    thresh_min_peak = 0.02

    if distribution[num-2] +thresh_monotonic < distribution[num -1] and distribution[num-1]+thresh_monotonic<distribution[0] \
        and distribution[0] > distribution[1]+thresh_monotonic and distribution[1] >distribution[2]+thresh_monotonic \
        and distribution[0] > thresh_min_peak:
        ridge.append(1)

        k +=1
    if distribution[num-1] + thresh_monotonic< distribution[0] and distribution[0] +thresh_monotonic < distribution[1] \
        and distribution[1] > distribution[2] +thresh_monotonic and distribution[2]>distribution[3]+thresh_monotonic\
        and distribution[1] > thresh_min_peak:
        ridge.append(2)
        k+=1


    for i in range(2, num-2):
        p1 = distribution[i-2]
        p2 = distribution[i-1]
        p3 = distribution[i]
        p4 = distribution[i+1]
        p5 = distribution[i+2]



        if p1+thresh_monotonic < p2 and p2 + thresh_monotonic < p3 and p3> p4+thresh_monotonic and p4 > p5+thresh_monotonic and p3 > thresh_min_peak:
            ridge.append(i+1)
            k+=1

    if distribution[num-4] + thresh_monotonic < distribution[num-3] and distribution[num-3]+thresh_monotonic <distribution[num-2]\
        and distribution[num-2]>distribution[num-1]+thresh_monotonic and distribution[num-1] >distribution[0] +thresh_monotonic \
        and distribution[num-2] >thresh_min_peak:
        ridge.append(num-1)
        k += 1

    if distribution[num-3] + thresh_monotonic < distribution[num-2] and distribution[num-2]+thresh_monotonic<distribution[num-1]\
        and distribution[num-1] > distribution[0] + thresh_monotonic and distribution[0] > distribution[1]+thresh_monotonic \
        and distribution[num-1] > thresh_min_peak:
        ridge.append(num)
        k += 1
    num_peak= k-1

    count = 1
    ridge_final = []

    if num_peak >= 3:
        for i in range(1,num_peak-1):
            start = int(np.floor((ridge[i-1]+ridge[i])/2))
            tail = int(np.floor((ridge[i]+ridge[i+1])/2))
            if distribution[start-1] <thresh_bright and distribution[tail-1] < thresh_bright and distribution[ridge[i]-1] - max(distribution[start-1], distribution[tail-1])>thresh_peak_height:
                ridge_final.append(ridge[i])
                count += 1

        start = int(np.floor((ridge[num_peak-1]+ridge[0]+num)/2))
        if start > num:
            start = start-num
        tail = int(np.floor((ridge[0]+ridge[1])/2))
        if distribution[start-1] < thresh_bright and distribution[tail-1] <thresh_bright and distribution[ridge[0]-1] - max(distribution[start-1], distribution[tail-1]) > thresh_peak_height:
            ridge_final.append(ridge[0])
            count += 1

        start = int(np.floor((ridge[num_peak-2]+ridge[num_peak-1])/2))
        tail = int(np.floor((ridge[num_peak-1]+ridge[0]+num)/2))
        if start > num:
            start = start - num
        if tail > num:
            tail = tail - num

        if distribution[start-1] < thresh_bright and distribution[tail-1] < thresh_bright and distribution[ridge[num_peak-1]-1]-max(distribution[start-1], distribution[tail-1]) > thresh_peak_height:
            ridge_final.append(ridge[num_peak-1])
            count += 1
    num_peak = len(ridge_final)

    if num_peak < 3:
        return isbranch, ridge

    peaks = [distribution[i-1, 0] for i in ridge_final]
    p_min = min(peaks)
    p_max = max(peaks)

    if not (p_min > thresh_min_peak and p_min/p_max>thresh_ratio_peaks):
        return isbranch, ridge
    isbranch = True







    
    return isbranch, ridge

if __name__ == "__main__":
    main("source/img_left1.bmp")
