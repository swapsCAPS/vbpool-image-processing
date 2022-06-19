# import the necessary packages
from lib import process_back
import numpy as np
import argparse
import cv2
import logging as log
import os

log.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
args = vars(ap.parse_args())
log.debug('args', args)

# load the input image and template from disk
log.debug("[INFO] loading images...")
image = cv2.imread(args["image"])

# align the images
process_back(image, debug=True)

