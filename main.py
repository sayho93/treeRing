import cv2
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

x1, x2, y1, y2 = 0, 0, 0, 0
tImg = None
draw = False


def onMouse(event, x, y, flags, paprm):
    global x1, x2, y1, y2, tImg, draw
    if draw:
        tImg = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        x1, x2, y1, y2 = 0, 0, 0, 0
        x1 = x
        y1 = y

    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(tImg, (x1, y1), (x, y), (255, 0, 0), 2)
        x2, y2 = x, y
        draw = False

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if draw:
            cv2.line(tImg, (x1, y1), (x, y), (255, 0, 0), 2)
            x2, y2 = x, y


def rotateImage(img, p1, p2):
    def _rotatePoint(p, rotationMatrix):
        return rotationMatrix.dot(np.array(p + (1,))).astype(int)
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        rotation_angle = 90
    else:
        rotation_angle = np.rad2deg(np.arctan((y1 - y2) / (x1 - x2)))

    rows, cols = img.shape[:2]
    center = (cols / 2, rows / 2)
    rotationMat = cv2.getRotationMatrix2D(center, rotation_angle, 1)

    abs_cos = abs(rotationMat[0, 0])
    abs_sin = abs(rotationMat[0, 1])

    new_w = int(rows * abs_sin + cols * abs_cos)
    new_h = int(rows * abs_cos + cols * abs_sin)

    rotationMat[0, 2] += new_w / 2 - center[0]
    rotationMat[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(img, rotationMat, (new_w, new_h))

    rx1, ry1 = _rotatePoint(p1, rotationMat)
    rx2, ry2 = _rotatePoint(p2, rotationMat)

    return rotated, rx1, ry1, rx2, ry2


def countRings(img, x1, y1, x2, y2):
    line = img[ry1 - 30: ry1 + 30, min(rx1, rx2):max(rx1, rx2)]
    gray = cv2.cvtColor(line[30:35, :, :], cv2.COLOR_BGR2GRAY)

    # #############################Image Processing
    # thresholding
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)

    # opening, closing
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # gaussian
    gaussian = cv2.GaussianBlur(closing, (5, 5), 0)
    # canny edge detecting
    res = cv2.Canny(gaussian, 120, 220)

    # sobelX = cv2.Sobel(closing, cv2.CV_64F, 1, 0, ksize=1)
    # sobelX = cv2.convertScaleAbs(sobelX)
    # sobelY = cv2.Sobel(closing, cv2.CV_64F, 0, 1, ksize=1)
    # sobelY = cv2.convertScaleAbs(sobelY)
    # res = cv2.addWeighted(sobelX, 1, sobelY, 1, 0)

    # res = cv2.Laplacian(closing, cv2.CV_8U)

    # cv2.imshow('gray', gray)
    # cv2.imshow('gaussian', gaussian)
    # cv2.imshow('threshold', threshold)
    # cv2.imshow('opening', opening)
    # cv2.imshow('closing', closing)
    # cv2.imshow('res', closing)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # #############################Image Processing

    line = cv2.rectangle(line, (0, 30), (line.shape[1], 35), (0, 0, 0), 1)
    n = 2
    b = [1.0 / n] * n
    a = 1.0
    m_res = res.mean(axis=0)
    f_intensity = signal.filtfilt(b, a, m_res)
    rings = (np.diff(np.clip(np.diff(f_intensity), -10, 0)) < -1)
    count = int((np.diff(rings) > 0).sum() / 2)

    return line, m_res, rings, count


def plotRings(img, intensity, rings, count):
    fig, (axim, axin, axring) = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))
    axim.imshow(img[..., ::-1], aspect="auto")
    axim.set_title("Cross section of the tree trunk where the ring are counted")
    axim.axis("off")
    axim.margins(0)
    axin.plot(intensity)
    axin.set_title("Average pixle intencity on the y axis inside the black box")
    axin.margins(0)
    x = np.arange(rings.shape[0])
    axring.fill_between(x, 0, rings)
    axring.set_title("Ring markers, total ring count is {}".format(count))
    axring.margins(0)
    plt.tight_layout()
    plt.show()


Tk().withdraw()
img_file = askopenfilename(filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
img = cv2.imread(img_file)

cv2.namedWindow("Tree rings", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Tree rings", onMouse)

tImg = img.copy()
while True:
    cv2.imshow("Tree rings", tImg)
    k = cv2.waitKey(30)
    if k & 0xFF == 27:
        cv2.destroyAllWindows()
        break

if (x1, y1) != (x2, y2):
    tImg = img.copy()

    # ################## TEST ####################
    gray = cv2.cvtColor(tImg, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    gaussian = cv2.GaussianBlur(closing, (5, 5), 0)
    res = cv2.Canny(gaussian, 120, 220)

    # sobelX = cv2.Sobel(closing, cv2.CV_64F, 1, 0, ksize=1)
    # sobelX = cv2.convertScaleAbs(sobelX)
    # sobelY = cv2.Sobel(closing, cv2.CV_64F, 0, 1, ksize=1)
    # sobelY = cv2.convertScaleAbs(sobelY)
    # res = cv2.addWeighted(sobelX, 1, sobelY, 1, 0)

    # res = cv2.Laplacian(closing, cv2.CV_8U)

    # cv2.imshow('gray', gray)
    # # cv2.imshow('gaussian', gaussian)
    # cv2.imshow('threshold', threshold)
    # cv2.imshow('opening', opening)
    # cv2.imshow('closing', closing)
    # cv2.imshow('res', closing)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # exit()
    # ################## TEST ####################

    rotated, rx1, ry1, rx2, ry2 = rotateImage(tImg, (x1, y1), (x2, y2))

    img_strip, intensity, rings_map, rings_count = countRings(rotated, rx1, ry1, rx2, ry2)
    plotRings(img_strip, intensity, rings_map, rings_count)
    print('The age of the tree:' + repr(rings_count) + ' years')