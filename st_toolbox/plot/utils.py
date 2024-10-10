import logging
from typing import List
import numpy as np



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)


def auto_split_contours(contour: np.ndarray, max_distance: float) -> List[np.ndarray]:
    return_list = []
    logger.debug("input contour shape {}".format(contour.shape))
    _tmp_a = np.array([contour[0]])
    for i in range(0, len(contour) - 1):
        if np.abs(np.linalg.norm(contour[i + 1] - contour[i])) < max_distance:
            _tmp_a = np.append(_tmp_a, [contour[i + 1]], axis=0)
        else:
            return_list.append(_tmp_a)
            _tmp_a = np.array([contour[i + 1]])
    return_list.append(_tmp_a)
    logger.debug("split into {} shapes".format(len(return_list)))
    return return_list

