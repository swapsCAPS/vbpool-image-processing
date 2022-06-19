import argparse
import cv2
import imutils
import numpy as np
import logging as log

def perspective_warp():
    aligned = cv2.warpPerspective(image, H, (w, h))

def request_user_input(image, debug=False):
    cv2.imshow("Select Left bla", imutils.resize(image, height=1080))
    cv2.waitKey(5000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")

    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    request_user_input(image)
