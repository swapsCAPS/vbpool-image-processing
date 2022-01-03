# import the necessary packages
import argparse
import cv2
import logging as log
import os
from glob import glob
from align_images import align_images
parse_num = lambda f: int(f.split("-")[-1].split(".")[0])

log.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to templage")
args = vars(ap.parse_args())
log.debug('args', args)

# load the input image and template from disk
log.debug("[INFO] loading images...")
#  image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

poule_file_names = glob("/home/dan/Downloads/poulespng/*")
print('poule_file_names', poule_file_names)

back_images = [file_name for file_name in poule_file_names if parse_num(file_name) % 2 == 0 ]
back_images = sorted(back_images, key=parse_num)

for back_image in back_images:
    num = parse_num(back_image)

    to_write = align_images(cv2.imread(back_image), template, max_features=1000, keep_percent=0.2, debug=False)

    cv2.imwrite(f"/tmp/poule_{num}_aligned.png", to_write)
