import argparse
import cv2
import numpy as np
import sys
import logging as log
import os
import imutils

log.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

def blur(image, amount=5, debug=False):
    return cv2.GaussianBlur(image, (amount, amount), 0)

def dilate(image, amount=5, iterations=1, debug=False):
    kernel = np.ones((amount,amount), np.uint8)
    return  cv2.dilate(image, kernel, iterations=iterations)

def erode(image, amount=5, iterations=1, debug=False):
    kernel = np.ones((amount,amount), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def threshold(image, amount=240):
    th, result = cv2.threshold(image, amount, 255, cv2.THRESH_BINARY)
    return result

def adaptive_threshold(image):
    print('image', image.shape)
    image_has_color = image.shape[2] > 2

    if image_has_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def invert(image):
    """
    Watch out with alpha channels, they'll also be inverted!
    """
    return 255 - image

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image that we'll align to template")
    ap.add_argument("-f", "--fn", required=True,
        help="function to run")
    ap.add_argument("-a", "--amount", type=int, required=False,
        help="function to run")

    args = vars(ap.parse_args())

    fn = args["fn"]
    image_name = args["image"]

    log.debug(f"args {args}")

    image = cv2.imread(image_name)

    del args["image"]
    del args["fn"]

    args = {k:v for (k,v) in args.items() if v is not None}

    log.debug(f"args {args}")

    result = globals()[fn](image, **args)

    cv2.imshow(fn, result)
    filename = f"{image_name}_{fn}.png"

    log.debug(f"writing file {filename}")

    cv2.imwrite(filename, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
