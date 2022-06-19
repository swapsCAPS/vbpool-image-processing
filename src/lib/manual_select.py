import argparse
from tkinter import W
import cv2
import imutils
import numpy as np
import logging as log
import math

def perspective_warp():
    aligned = cv2.warpPerspective(image, H, (w, h))

def request_user_input(image, debug=False):
    global inputs
    clone = image.copy()
    inputs = []

    steps = [
        dict(title="Select TL"),
        dict(title="Select TR"),
        dict(title="Select BR"),
        dict(title="Select BL"),
    ]

    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            inputs.append([x, y])
            print(x, y)

    cv2.namedWindow("Select points")
    cv2.setMouseCallback("Select points", click_handler)

    while len(inputs) < 4:
        cv2.imshow("Select points", imutils.resize(image, height=1080))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
        elif key == ord("q"):
            break

    print("inputs:")
    print("TL", inputs[0])
    print("TR", inputs[1])
    print("BR", inputs[2])
    print("BL", inputs[3])

    input=np.float32(inputs)

    width = round(math.hypot(input[0,0]-input[1,0], input[0,1]-input[1,1]))
    height = round(math.hypot(input[0,0]-input[3,0], input[0,1]-input[3,1]))

    x = input[0,0]
    y = input[0,1]

    hh, ww = image.shape[:2]

    output = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])

    matrix = cv2.getPerspectiveTransform(input, output) 
    aligned = cv2.warpPerspective(image, matrix, (ww, hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
    while True:
        cv2.imshow("Select points", imutils.resize(aligned, height=1080))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")

    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    request_user_input(image)
