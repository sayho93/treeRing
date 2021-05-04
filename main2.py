import numpy as np
import cv2
import sys

frame = cv2.imread("rings_slice.jpg")

if frame is None:
    print('Error loading image')
    exit()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

rows = frame.shape[0]
cols = frame.shape[1]

# Use centre column
column_index = cols / 2

ring_count = 0;

# Start with the second row
for i in range(1, rows):
    # If this pixel is white and the previous pixel is black
    if 255 == frame[i, column_index] and 0 == frame[i - 1, column_index]:
        ring_count += 1;

print(ring_count)
