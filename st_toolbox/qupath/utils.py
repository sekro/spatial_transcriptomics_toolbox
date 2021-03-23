"""

"""
from typing import List, Tuple, Dict

import numpy as np
import cv2
import json
from .coregistration import CoRegistrationData

def fast_TRS_2d(input, transform_matrix, input_is_point=False):
    """
    Fast translation, rotation & scale in 2D using np.einsum in case input is not a single point
    :param input: point or point lst as np.array
    :param transform_matrix: np.array
    :param input_is_point: bool
    :return: transformed point or list of points
    """
    if input_is_point:
        return np.delete(np.dot(transform_matrix, np.insert(input, 2, 1)), 2)
    else:
        return np.delete(np.einsum('jk,ik->ij', transform_matrix, np.insert(input, 2, 1, axis=1)), 2, 1)


def make_mask(shape, contour):
    """
    Binary mask from cv2 styled contour (gets filled)
    :param shape: dim of the mask
    :param contour: inpout contour
    :return: the binary image / mask
    """
    mask = np.zeros(shape, np.int32)
    cv2.drawContours(mask, [contour], 0, (255), -1)
    return mask


def load_co_registration_data_from_json(filename: str) -> Dict[str, CoRegistrationData]:
    """
    Load saved output from QuPath img import & processing function
    basically just a stupid wrapper for json.load for now
    :param filename: the JSON file with the data
    :return: data dictionary
    """
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    co_reg_data = {}
    for index, data in data.items():
        co_reg_data[index] = CoRegistrationData(
            name=str(data['name']),
            target_w=int(data['target_w']),
            target_h=int(data['target_h']),
            transform_matrix=np.array(data['transform_matrix']),
            moving_img_name=str(data['moving_img_name'])
        )
    return co_reg_data


def save_co_registration_data_to_json(datasets: Dict[str, CoRegistrationData], output_file) -> str:
    """
    Save output from QuPath img import & processing function to JSON file
    Warning: Currently a bit stupid - does not check if output_path ok
    :param datasets: the data sets to be saved
    :param output_file: path to output JSON-file
    :return: the JSON string saved to the JSON file
    """
    to_json_tmp = {}
    for index, data in datasets.items():
        to_json_tmp[index] = {
            'name': data.name,
            'target_w': data.target_w,
            'target_h': data.target_h,
            'transform_matrix': data.transform_matrix.tolist(),
            'moving_img_name': data.moving_img_name
        }
    with open(output_file, "w") as data_file:
        json.dump(to_json_tmp, data_file, indent=4, sort_keys=True)
    return json.dumps(to_json_tmp, indent=4, sort_keys=True)


def coreg_qp_hes(in_qp_hes_path: str, out_qp_hes_path: str, coregdata: CoRegistrationData) -> bool:
    moving_img = cv2.imread(in_qp_hes_path, cv2.IMREAD_COLOR)
    transformed_img = cv2.warpPerspective(moving_img, coregdata.transform_matrix,
                                          (coregdata.target_w, coregdata.target_h))
    cv2.imwrite(out_qp_hes_path, transformed_img)
    return True
