###
# Following tutorial at https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
###

import numpy as np
import imutils
import cv2
import logging as log
import os
from glob import glob

from lib.align_images import align_images
from lib.rect_utils import (
    snap_rects_to_nearest_vertical,
    align_rects_to_self,
    cluster_rects_on_diff,
    to_cv2_rect,
    find_missing_rects,
)

log.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

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

    left_table_coords = [[.0, .270], [0.492, .845]]
    right_table_coords = [[.495, .270], [0.992, .845]]

    left_table = crop(image, left_table_coords, ratio_crop=True)
    right_table = crop(image, right_table_coords, ratio_crop=True)

    if debug is True:
        stacked = np.hstack([left_table])#, right_table])
        cv2.imshow("cropped", stacked)
        cv2.waitKey(0)

    return dict(
        left=left_table,
        right=right_table,
    )

def find_rects(image, debug=False):
    img_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    th, working_img = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
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

def dedupe_rects(input_rects, eps):
    """
    Accepts array with [Rect, Rect] i.e. cv2.rectangle()

    Args:
        input_rects ([Rect]) [cv2.rectangle(), cv2.rectangle(), ...]
        eps (int)

    Returns:
        [Rect]
    """

    working_rects = []
    for r in input_rects:
        working_rects.append(r)
        working_rects.append(r) # lol do it twice ;'D

    result, weights = cv2.groupRectangles(working_rects, 1, eps)

    log.debug(f"dedupe_rects {len(result)} / {len(working_rects)}")

    return result

MIN_ROW_HEIGHT_PX = 48
MAX_ROW_HEIGHT_PX = 75

MIN_STANCE_WIDTH_PX = 120
MAX_STANCE_WIDTH_PC = 165

MIN_TOTO_WIDTH_PX = 63
MAX_TOTO_WIDTH_PX = 82

FIRST_HALF_MAX_X_THRESHOLD_PX = 50
SECOND_HALF_MIN_X_THRESHOLD_PX = 180
SECOND_HALF_MAX_X_THRESHOLD_PX = 230
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

            if x > SECOND_HALF_MIN_X_THRESHOLD_PX and x < SECOND_HALF_MAX_X_THRESHOLD_PX:
                log.debug('found second half')
                second_half.append(r)
                continue

        elif w > MIN_TOTO_WIDTH_PX and w < MAX_TOTO_WIDTH_PX and x > TOTO_MIN_X_THRESHOLD_PX:
            log.debug('found toto')
            toto.append(r)
            continue

        log.warn(f"found nothing for {r} {w} {h}")

    process = lambda x: cluster_rects_on_diff(x, axis=1, min_diff=30)
    first_half  = process(first_half)
    second_half = process(second_half)
    toto        = process(toto)

    process = lambda x: align_rects_to_self(x, axis=1, method="average", dedupe=True)
    first_half  = [process(group) for group in first_half]
    second_half = [process(group) for group in second_half]
    toto        = [process(group) for group in toto]

    first_half = [rs[0] for rs in first_half]
    second_half = [rs[0] for rs in second_half]
    toto = [rs[0] for rs in toto]

    second_half = find_missing_rects(image, second_half, debug=True)

    frame = image.copy()
    first_half  = [to_cv2_rect(frame, r, color=(255, 0, 0), thickness=3) for r in first_half]
    second_half = [to_cv2_rect(frame, r, color=(0, 255, 0), thickness=3) for r in second_half]
    toto        = [to_cv2_rect(frame, r, color=(0, 0, 255), thickness=3) for r in toto]

    if debug is True:
        debug_image = frame.copy()

        log.debug(f"first_half:  {len(first_half)}")
        log.debug(f"second_half: {len(second_half)}")
        log.debug(f"toto:        {len(toto)}")

        debug_image = imutils.resize(debug_image, height=1200)

        cv2.imshow("rects", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    first_half = np.array(first_half)
    second_half = np.array(second_half)
    toto = np.array(toto)

    """
    We have a list of rects looking like [ [ x1, y1, x2, y2 ], ... ]
    We want to align them on the x and y axis based on x1 and y1
    At this point we assume that the rects are of the correct dimensions,
    but we could also take an average of those and give each rect the same dimensions


    """

def find_stances_and_toto_naive(
    image,
    debug=False,
    template = cv2.imread("./pool-form-back-left.png"),
    first_half_x_pos = 807,
    second_half_x_pos = 972,
    toto_x_pos = 1136,
    row_count = 26
):
    aligned = align_images(image, template, debug=False)

    match_width = 155
    toto_width = 70

    row_height = 62

    first_half = []
    second_half = []
    toto = []

    top_offset = 10
    row_y_diff = 77

    for row in range(0, row_count):
        y_pos = top_offset + (row * row_y_diff)

        first_half.append([
            first_half_x_pos,
            y_pos,
            first_half_x_pos + match_width,
            y_pos + row_height,
        ])

        second_half.append([
            second_half_x_pos,
            y_pos,
            second_half_x_pos + match_width,
            y_pos + row_height,
        ])

        toto.append([
            toto_x_pos,
            y_pos,
            toto_x_pos + toto_width,
            y_pos + row_height,
        ])

    frame = aligned.copy()
    first_half  = [to_cv2_rect(frame, r, color=(255, 0, 0), thickness=3) for r in first_half]
    second_half = [to_cv2_rect(frame, r, color=(0, 255, 0), thickness=3) for r in second_half]
    toto        = [to_cv2_rect(frame, r, color=(0, 0, 255), thickness=3) for r in toto]

    return frame

    #  cv2.imshow("debug_image", frame)
    #  cv2.waitKey(0)
    #  cv2.destroyAllWindows()

parse_num = lambda f: int(f.split("-")[-1].split(".")[0])

def process_back(image, debug=False):
    poule_file_names = glob("poulepngs/*")
    print('poule_file_names', poule_file_names)

    back_images = [file_name for file_name in poule_file_names if parse_num(file_name) % 2 == 0 ]
    back_images = sorted(back_images, key=parse_num)

    template = cv2.imread("./pool-form-back-1.png")
    for back_image in back_images:
        num = parse_num(back_image)

        image = cv2.imread(back_image)

        aligned = align_images(image, template, debug=False)

        back = crop_back(aligned, debug=False)

        left = find_stances_and_toto_naive(
            back["left"],
            template = cv2.imread("./pool-form-back-left.png"),
            debug=False,
            first_half_x_pos = 807,
            second_half_x_pos = 972,
            toto_x_pos = 1136,
            row_count = 26
        )

        right = find_stances_and_toto_naive(
            back["right"],
            template = cv2.imread("./pool-form-back-right.png"),
            debug=False,
            row_count=25,
            first_half_x_pos=798,
            second_half_x_pos=966,
            toto_x_pos=1133,
        )

        cv2.imwrite(f"/tmp/poule_{num}_left.png", left)
        cv2.imwrite(f"/tmp/poule_{num}_right.png", right)



    #  find_stances_and_toto(back["left"], debug=True)
    #  find_stances_and_toto(back["right"], debug=True)



