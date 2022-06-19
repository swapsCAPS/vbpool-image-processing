import argparse
from tkinter import W
import cv2
import imutils
import numpy as np
import logging as log
import math

def perpective_input(source, debug=False):
    global inputs

    source_height, source_width = source.shape[:2]

    req_height = 1000

    ratio = req_height / source_height
    print(ratio)

    image = imutils.resize(source, height=req_height)
    clone = image.copy()
    inputs = []

    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            inputs.append([x, y])
            print(x, y)

    cv2.namedWindow("Select points")
    cv2.setMouseCallback("Select points", click_handler)

    while len(inputs) < 4:
        for p in inputs:
            cv2.drawMarker(image, p, (0, 0, 255), cv2.MARKER_CROSS)

        cv2.imshow("Select points", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
        if key == ord("u"):
            image = clone.copy()
            if len(inputs) > 0: inputs.pop()
            continue
        elif key == ord("q"):
            break

    input=np.float32(inputs)

    width = round(math.hypot(input[0,0]-input[1,0], input[0,1]-input[1,1]))
    height = round(math.hypot(input[0,0]-input[3,0], input[0,1]-input[3,1]))

    x = input[0,0]
    y = input[0,1]

    output = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])

    matrix = cv2.getPerspectiveTransform(input, output)
    return cv2.warpPerspective(source, matrix, (source_width, source_height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")

    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    aligned = perpective_input(image)

    while True:
        cv2.imshow("Select points", aligned)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break