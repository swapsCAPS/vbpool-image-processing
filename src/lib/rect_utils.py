import numpy as np
import cv2
import logging as log
import os

log.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

def cluster_rects_on_diff(input_rects, axis=0, min_diff=30):
    """
    Utility function to group rectangles by the difference in distance from each other.

    Args:
        input_rects (list) [ [ x1, y1, x2, y2 ], [ x1, y1, x2, y2 ] ]
        axis (int) 0 for x, 1 for y, this is indicating on what index of [ x1, y1... ] we're operating
        min_diff (int) if points are further away than this distance, they will be regarded as a separate group

    Returns:
        On `axis` sorted array of groups of rects e.g. [ [ [x1,x2,y1,y2], [x1,x2,y1,y2] ], [x1,x2,y1,y2] ]
    """

    # First sort the rects by axis
    input_rects = sorted(input_rects, key=lambda r: r[axis])
    input_rects = np.array(input_rects)

    # Get the difference for each axis pos with [..., axis]
    diff_indexes = np.where(np.diff(input_rects[..., axis]) > min_diff)[0]
    # np.split needs + 1
    diff_indexes += 1

    log.debug(f"diff_indexes {diff_indexes}")

    # split original array on the indexes we've fetched
    return np.split(input_rects, diff_indexes)

def align_rects_to_self(input_rects, axis=0, method="average"):
    input_rects = np.array(input_rects)
    print('input_rects', input_rects)

    positions = input_rects[..., axis] # Grab axis from each array

    wanted_pos = int(getattr(np, method)(positions))

    result = []
    for [x1, y1, x2, y2] in input_rects:
        working_rect = [x1, y1, x2, y2]

        translate = wanted_pos - working_rect[axis]

        working_rect[axis]     = working_rect[axis]     + translate
        working_rect[axis + 2] = working_rect[axis + 2] + translate

        result.append(working_rect)

    log.debug(result)

    return result

def snap_rects_to_nearest_vertical(rects, to_nearest=5):
    result = []
    for [x1, y1, x2, y2] in rects:
        diff = y1 % to_nearest
        y1 = y1 - diff
        y2 = y2 - diff

        result.append([ x1, y1, x2, y2 ])

    return result

def align_rects_vertical(rects, method="average"):
    """
    Accepts array with [x1, y1, x2, y2] i.e. [tl_x, tl_y, br_x, br_y]

    :param rects [[int, int, int, int], [int, int, int, int]]
    """
    rects = np.array(rects)

    max_diff = 50
    x_positions = rects[..., 0] # Grab first item from each array

    diff = np.max(x_positions) - np.min(x_positions)

    log.debug(f"align_rects x1: {diff}")

    if diff > max_diff:
        raise Exception(f"align_rects() diff too large x1: '{diff}' > {max_diff}px")

    avg_x1 = int(getattr(np, method)(x_positions))

    result = []
    for [x1, y1, x2, y2] in rects:
        translate = avg_x1 - x1
        x1 = x1 + translate
        x2 = x2 + translate

        result.append([ x1, y1, x2, y2 ])

    log.debug(result)

    return result

def to_cv2_rect(image, rect, color=(255, 0, 0), thickness=3):
   """
   Args:
       image (ndarray) image to draw the rectangle on
       rect ([int, int, int, int]) [x1, y1, x2, y2] i.e. [tl_x, tl_y, br_x, br_y]
       color (int, int, int) blue, green, red
       thickness (int) line width

   Returns:
       cv2.rectangle()
   """

   [x1, y1, x2, y2] = rect

   return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

