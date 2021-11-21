import cv2
import numpy as np
import logging as log
import imutils

def align_images(image, template, max_features=1000, keep_percent=0.4, debug=False):
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
        debug_img = imutils.resize(debug_img, width=1400)
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

    if debug is True:
        cv2.imshow("Aligned image", imutils.resize(aligned, height=900))
        cv2.waitKey(0)

    log.debug("done!")
    return aligned

