import cv2
import numpy as np
import argparse
from cv2.ximgproc import thinning
from itertools import combinations

def getCropLoc(img):    #픽셀 값이 21 이상되는 값이 3번 반복되는 것을 찾음
    pprev = 5000
    prev = 4000
    now = 0
    i = 0
    while (prev != now) or now == 0 or (pprev != prev):
        pprev = prev
        prev = now


        now = np.sum(img[i, :15]>21)
        i += 1
    first = i-2
    pprev = 5000
    prev = 4000
    now = 0
    i = img.shape[0]-3

    while (prev != now) or now == 0 or (pprev != prev):
        pprev = prev
        prev = now

        now = np.sum(img[i, :15] > 28)
        i -= 1
    second = i+3

    pprev = 5000
    prev = 4000
    now = 0

    i = 0

    while (prev != now) or now == 0 or (pprev != prev):
        pprev = prev
        prev = now

        now = np.sum(img[:15, i]>28)
        i += 1
    third = i-2

    pprev = 5000
    prev = 4000
    now = 0
    i = img.shape[1]-100

    while (prev != now) or now == 0 or (pprev != prev):
        pprev = prev
        prev = now

        now = np.sum(img[:8, i]>28)
        i -= 1
    fourth = i+3

    return first, second, third, fourth




def hist_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)

    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img2 = cdf[img]

    return img2

def get_distance(p1, p2):   #피타고라스 
    return np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))


def getHoughDist(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape

    x = [width // 3]
    x.append(x[0] * 2)
    y = [height // 3]
    y.append(y[0] * 2)

    empty = np.zeros_like(img)

    pts = np.zeros((0, 3, 2), dtype = np.int)
    pts = np.append(pts, np.array([[0, 0], [x[0], 0], [0, y[0]]]).reshape(-1, 3, 2), axis = 0)
    pts = np.append(pts, np.array([[0, y[1]], [0, height - 1], [x[0], height - 1]]).reshape(-1, 3, 2), axis =  0)
    pts = np.append(pts, np.array([[x[1], 0], [width - 1, 0], [width - 1, y[0]]]).reshape(-1, 3, 2), axis = 0)
    pts = np.append(pts, np.array([[x[1], height - 1], [width - 1, height - 1], [width - 1, y[1]]]).reshape(-1, 3, 2), axis = 0)

    for p in pts:
        cv2.fillPoly(empty, [p], (255, 255, 255), cv2.LINE_AA)

    img = cv2.bitwise_and(img, empty)
    invGamma = 1.5
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype('uint8')

    gamma_corrected = cv2.LUT(img, table)
    radius2 = max(height, width)
    circles = cv2.HoughCircles(gamma_corrected, cv2.HOUGH_GRADIENT, 1, 40, param1=70, param2=25, minRadius=radius2 // 2 - 20,
                               maxRadius=radius2 // 2 + 10)
    circles = np.uint16(np.around(circles[0]))

    index = np.argmax(circles[:, 2])


    return np.max(circles[:, 2])*2, circles[index, :2]

def getSegDist(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 30, param2 = 15, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    index = np.argmax(circles[0][:, 2])
    circles = circles[0][index]

    return circles[2]*2, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def getbranch(img, starts):
    points = []
    way = [(-1, -1), (-1, 0), (-1, 1), (0, -1,), (0, 1), (1, -1), (1, 0), (1, 1)]
    for s in starts:
        stack = []
        x, y = s[1], s[0]
        stack.append((x, y))
        pop = 0
        branch = []
        while len(stack) != 0:
            img[x, y] = 0

            for w in way:
                if img[x + w[0], y + w[1]] != 0:
                    if pop == 1:
                        branch.append((x, y))
                    x = x + w[0]
                    y = y + w[1]
                    pop = 0
                    stack.append((x, y))
                    break
            else:
                x, y = stack.pop()
                pop = 1
        points.append(branch)
    return points[0]

def getPoints(points, gamma_corrected):
    dicts = {}
    for p in points:
        img = np.zeros_like(gamma_corrected)
        cv2.circle(img, (p[1], p[0]), 30, 255, -1)
        sums = cv2.bitwise_and(img, gamma_corrected)
        sums = np.sum(sums!= 0)
        dicts[sums] = p
    dicts = sorted(dicts.items(), key = lambda x:x[0], reverse = True)[:4]
    points = [p for _, p in dicts]

    combi = list(combinations(points, 2))

    dist_dict = {}

    for p1, p2 in combi:
        dist = get_distance(p1, p2)
        dist_dict[dist] = (p1, p2)
    line_point = sorted(dist_dict.items(), key = lambda x: x[0], reverse=True)[0]
    line_distance = line_point[0]
    line_point = line_point[1]

    return points, line_distance, line_point



def getLineDist(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.imread(path)

    first, second, third, fourth = getCropLoc(gray)
    gray[:first, :] = 255
    gray[second:, :] = 255
    gray[:, :third] = 255
    gray[:, fourth:] = 255

    invGamma = 1.5
    table = np.array([((i/255.0)**invGamma)*255 for i in range(0, 256)]).astype('uint8')

    for i in range(2):
        gray = hist_equalization(gray)
        gray = cv2.LUT(gray, table)
    gray = hist_equalization(gray)
    gray = cv2.fastNlMeansDenoising(gray.astype('uint8'), None, 3, 7, 21)

    _, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    gamma = gray.copy()

    gray = thinning(gray, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    contour, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    maxi = 0
    max_index = 0
    starts = []
    for i, cnt in enumerate(contour):
        if len(cnt) >maxi:
            maxi = len(cnt)
            max_index = i
    cnt = contour[max_index]
    starts.append(tuple(cnt[cnt[:, :, 0].argmin()][0]))
    points = getbranch(gray.copy(), starts)
    pp = points.copy()

    point, line_distance, line_point = getPoints(points, gamma)

    cv2.line(orig, (line_point[0][1], line_point[0][0]), (line_point[1][1], line_point[1][0]), (0, 0, 255), 3)

    return line_distance, orig, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), pp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help = "img path", type = str)
    parser.add_argument("-spath", help = 'segmentation path', type = str)
    args = parser.parse_args()

    if not args.path or not args.spath:
        parser.error("no path")

    cap, center = getHoughDist(args.path)
    seg, seg_img = getSegDist(args.spath)
    line, orig, thin, points = getLineDist(args.path)

    real_line = (line*14)/cap
    real_seg = (real_line*seg)/line

    orig = cv2.bitwise_xor(seg_img, orig)
    orig = cv2.bitwise_xor(orig, thin)


    for p in points:
        cv2.circle(orig, (p[1], p[0]), 2, (0, 255, 0), 5)

    cv2.circle(orig, tuple(center), cap//2, (0, 255, 0), 5)
    cv2.imwrite("result.png", orig)





    print("%s mm"%real_seg)




