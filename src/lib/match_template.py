import argparse
import cv2
import numpy as np
import logging as log

from preprocessing import add_alpha
from rect_utils import to_cv2_rect

def simpleMatchTemplate(
    image: np.ndarray,
    template: np.ndarray,
    threshold=0.85,
    norm=True,
    debug=False,
):
    """
    NOTE Alpha channels are tricky.
         And IIRC matchTemplate only works when the shape of the template and the image are the same.
         using cv2.imread always imports in BGR! Use cv2.IMREAD_UNCHANGED for source color format
         Using `pngcheck` cli tool is helpful here just use pngcheck file.png for color info

    Now what will be a nice approach here. We could convert all input images to alpha, but we could also
    just add it, if it does not exist using numpy

    """
    if image is None:
        raise Exception("Please provide an image")

    if template is None:
        raise Exception("Please provide an template")

    log.debug(f"image.shape:     {image.shape}")
    log.debug(f"template.shape:  {template.shape}")

    if image.shape[2] != 4: # TODO too naive for grayscale
        image = add_alpha(image)
        log.info(f"added alpha channel to image {image.shape}")
    if template.shape[2] != 4:
        template = add_alpha(template)
        log.info(f"added alpha channel to template {template.shape}")

    alpha_mask = np.array(cv2.split(template)[3])

    log.debug(f"alpha_mask.shape: {alpha_mask.shape}")

    res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED, mask=alpha_mask)

    log.debug('res', res)

    # Normalize from 0 to 1
    if norm:
        log.debug(f"normalizing")
        res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX, -1)

    match_locations = np.where(res >= threshold)

    log.debug('match_locations', match_locations)

    h, w = template.shape[:2]

    rects = []
    for (x, y) in zip(*match_locations[::-1]):
        x1 = int(x)
        y1 = int(y)
        x2 = x + w
        y2 = y + h
        rects.append([ x1, y1, x2, y2 ])

    return rects

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")

    ap.add_argument("-tmpl", "--template", required=True,
        help="path to template that we'll use with cv2.matchTemplate()")

    ap.add_argument("-th", "--threshold", type=float, required=False,
        help="threshold for matches")

    args = vars(ap.parse_args())

    image_name = args["image"]
    template_name = args["template"]

    log.debug(f"args {args}")

    # NOTE image will have color scheme BGR, even if grayscale
    image = cv2.imread(image_name)

    # NOTE expecting template with alpha channel
    template = cv2.imread(template_name, cv2.IMREAD_UNCHANGED)

    del args["image"]
    del args["template"]

    args = {k:v for (k,v) in args.items() if v is not None}

    log.debug(f"args {args}")

    # TODO more fns?
    result = globals()["simpleMatchTemplate"](image, template, **args)

    [to_cv2_rect(image, r, color=(0, 0, 255), thickness=3) for r in result]

    cv2.imshow(f"hi", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
