###
# Following tutorial at https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
###

import numpy as np
import imutils
import cv2
import logging as log
import os

log.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

def align_images(image, template, max_features=500, keep_percent=0.2, debug=False):
    image_gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    log.debug("ORB_create")
    orb = cv2.ORB_create(max_features)

    log.debug("detectAndCompute")
    (keypoints_a, descriptors_a) = orb.detectAndCompute(image_gray, None)
    (keypoints_b, descriptors_b) = orb.detectAndCompute(template_gray, None)

    log.debug("DescriptorMatcher_create")
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    log.debug("matcher.match")
    matches = matcher.match(descriptors_a, descriptors_b, None)

    log.debug("sorting")
    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keep_percent)
    log.debug(f"Keeping {keep} matches")
    matches = matches[:keep]

    if debug is True:
        log.debug("rendering debug image")
        debug_img = cv2.drawMatches(image, keypoints_a, template, keypoints_b, matches, None)
        debug_img = imutils.resize(debug_img, width=1000)
        cv2.imshow("Matched keypoints", debug_img)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    log.debug("allocating memory")
    points_a = np.zeros((len(matches), 2), dtype="float")
    points_b = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        points_a[i] = keypoints_a[m.queryIdx].pt
        points_b[i] = keypoints_b[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    log.debug("findHomography")
    (H, mask) = cv2.findHomography(points_a, points_b, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    log.debug("warpPerspective")
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image

    log.debug("done!")
    return aligned

def crop(image, bounds, ratio_crop=False):
    log.debug(f"crop() {image.shape}")
    [ tl, br ] = bounds
    width = image.shape[1]
    height = image.shape[0]
    if ratio_crop:
        tl[0] = int(tl[0] * width)
        tl[1] = int(tl[1] * height)
        br[0] = int(br[0] * width)
        br[1] = int(br[1] * height)

    # [rows, columns]
    return image[tl[1]:br[1], tl[0]:br[0]]

def crop_back(back_image, debug=False):
    image = back_image.copy()

    left_table_coords = [[.318, .270], [0.492, .843]]
    right_table_coords = [[.819, .270], [0.992, .843]]

    left_table = crop(image, left_table_coords, ratio_crop=True)
    right_table = crop(image, right_table_coords, ratio_crop=True)

    if debug is True:
        stacked = np.hstack([left_table, right_table])
        cv2.imshow("cropped", stacked)
        cv2.waitKey(0)

    return dict(
        left=left_table,
        right=right_table,
    )

def find_rects(image, debug=False):
    img_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    th, working_img = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
    working_img = 255 - working_img

    #  working_img=cv2.GaussianBlur(working_img, (3, 3), 1)

    contours, hierarchy = cv2.findContours(
        image=working_img,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    rects = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        x, y, w, h = cv2.boundingRect(approximation)
        rects.append((x, y, w, h))

    return rects

MIN_ROW_HEIGHT_PX = 48
MAX_ROW_HEIGHT_PX = 75

MIN_STANCE_WIDTH_PX = 135
MAX_STANCE_WIDTH_PC = 165

MIN_TOTO_WIDTH_PX = 63
MAX_TOTO_WIDTH_PX = 82

FIRST_HALF_MAX_X_THRESHOLD_PX = 30
TOTO_MIN_X_THRESHOLD_PX = 320

def find_stances_and_toto(image, debug=False):
    log.debug(f"MIN_ROW_HEIGHT_PX {MIN_ROW_HEIGHT_PX}")
    log.debug(f"MAX_ROW_HEIGHT_PX {MAX_ROW_HEIGHT_PX}")

    rects = find_rects(image, debug=False)

    first_half = []
    second_half = []
    toto = []
    all_rects = []
    for (x, y, w, h) in rects:
        # If the rect is too high don't do anything
        if h < MIN_ROW_HEIGHT_PX or h > MAX_ROW_HEIGHT_PX: continue

        r = [x, y, x + w, y + h]

        all_rects.append(r)

        if w > MIN_STANCE_WIDTH_PX and w < MAX_STANCE_WIDTH_PC:
            if x < FIRST_HALF_MAX_X_THRESHOLD_PX:
                log.debug('found first half')
                first_half.append(r)
                continue

            log.debug('found second half')
            second_half.append(r)
            continue

        elif w > MIN_TOTO_WIDTH_PX and w < MAX_TOTO_WIDTH_PX and x > TOTO_MIN_X_THRESHOLD_PX:
            log.debug('found toto')
            toto.append(r)
            continue

        log.warn(f"found nothing for {r}")

    print('all_rects', all_rects)
    print('first_half', first_half)
    print('toto', toto)


    if debug is True:
        debug_image = image.copy()
        #  debug_image = image.copy()
        for r in first_half:
           cv2.rectangle(debug_image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 3)
        for r in second_half:
           cv2.rectangle(debug_image, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 3)
        for r in toto:
           cv2.rectangle(debug_image, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 3)

        cv2.imshow("rects", debug_image)
        cv2.waitKey(0)

def process_back(image, debug=False):
    template = cv2.imread("./pool-form-back-1.png")
    aligned = align_images(image, template, debug=False)

    back = crop_back(aligned, debug=False)

    find_stances_and_toto(back["left"], debug=True)
    find_stances_and_toto(back["right"], debug=True)



def parse_front():
    log.debug("parse_front")
