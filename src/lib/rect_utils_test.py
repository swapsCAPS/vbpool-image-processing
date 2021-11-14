import numpy as np
from lib.rect_utils import cluster_rects_on_diff, to_cv2_rect

def test_snap_rects_to_nearest_vertical():
    fixture = [
        [0,  0, 10, 10],
        [0,  2, 13, 12],
        [0,  1, 11, 11],
        [0, 16, 10, 26],
        [0, 17, 10, 27],
        [0, 15, 10, 25],
        [0, 20, 10, 30],
        [0, 22, 10, 32],
        [0, 20, 10, 30],
        [0, 21, 10, 31],
    ]

    result = cluster_rects_on_diff(fixture, axis=1, min_diff=2)
    expected = [
        np.array([
            [0,  0, 10, 10],
            [0,  1, 11, 11],
            [0,  2, 13, 12],
        ]),
        np.array([
            [0, 15, 10, 25],
            [0, 16, 10, 26],
            [0, 17, 10, 27],
        ]),
        np.array([
            [0, 20, 10, 30],
            [0, 20, 10, 30],
            [0, 21, 10, 31],
            [0, 22, 10, 32],
        ]),
    ]

    print('result', result)
    print('expected', expected)

    """
    THIS FUCKING FAILS BUT ARRAYS ARE THE SAME WTF?!
    """
    np.testing.assert_array_equal(result, expected)

