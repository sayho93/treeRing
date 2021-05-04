import numpy as np
import cv2
import sys

frame = cv2.imread("rings_slice.jpg")

if frame is None:
    print('Error loading image')
    exit()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

gray = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

rows = frame.shape[0]
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                          param1=60, param2=25,
                          minRadius=1, maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))
    count = 0
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(frame, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(frame, center, radius, (255, 0, 255), 3)
        count = count + 1;
print('The age of the tree:' + repr(count) + ' years')

# cv.imshow("detected circles", src)
cv2.waitKey(0)