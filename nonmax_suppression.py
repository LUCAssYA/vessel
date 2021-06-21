import numpy as np

def nonmax_suppresion(Ivessel, connected_component, connected_peaks, distance_closet_feature_point, num_smallest_component):
    num_connected_component = len(connected_component)

    centers = np.array([])

    for i in range(num_connected_component):
        tmpx = 0
        tmpy = 0
        weight= 0

        for j in range(connected_component[i].shape[0]):
            if len(connected_component[i].shape) == 2:
                x = int(connected_component[i][j, 0])
                y = int(connected_component[i][j, 1])
            else:
                x = int(connected_component[i][0])
                y = int(connected_component[i][1])
            tmpx = tmpx+x*Ivessel[y-1, x-1]
            tmpy = tmpy+y*Ivessel[y-1, x-1]
            weight = weight+Ivessel[y-1, x-1]
        temp = np.array([tmpx/weight, tmpy/weight, connected_component[i].shape[0]]) if len(connected_component[i].shape) ==2 else np.array([tmpx/weight, tmpy/weight, 1])
        centers = np.vstack((centers, temp)) if centers.size != 0 else temp

    for i in range(centers.shape[0]):
        for j in range(i+1, centers.shape[0]):
            if centers[i, 2] < centers[j, 2]:
                temp = centers[i, :].copy()
                centers[i, :] = centers[j, :].copy()
                centers[j, :] = temp
                temp = connected_peaks[i].copy()
                connected_peaks[i] = connected_peaks[j].copy()
                connected_peaks[j] = temp
    k = 0
    feature_points = np.array([])
    ridges = {}

    for i in range(num_connected_component):
        flag_too_close = False
        for j in range(i):
            aa = np.linalg.norm(centers[i, :2]-centers[j,  :2])
            if np.linalg.norm(centers[i, :2]-centers[j,  :2]) < distance_closet_feature_point:
                flag_too_close = True
                continue

        if flag_too_close == False:
            feature_points = np.vstack((feature_points, centers[i,  :])) if feature_points.size != 0 else centers[i, :]
            ridges[k] = connected_peaks[i].copy()
            k += 1

    index_goodfeature = np.array(feature_points[:, 2] >= num_smallest_component, dtype = np.bool)
    idx = np.where(index_goodfeature==True)[0]
    feature_points = feature_points[idx, :]

    k = 0
    ridges_final = {}

    for i in range(len(index_goodfeature)):
        if index_goodfeature[i] == True:
            ridges_final[k] = ridges[i].copy()
            k+= 1
    ridges = ridges_final.copy()

    return feature_points, ridges