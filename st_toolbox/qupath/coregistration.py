"""
image co-registration functions - part of spatial transcriptomics toolbox
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""
from typing import List

import numpy as np
import cv2
import os
import concurrent.futures
from dataclasses import dataclass


@dataclass
class CoRegistrationData:
    name: str
    target_w: int
    target_h: int
    transform_matrix: np.array
    moving_img_name: str

def register_imgs(moving_img, target_img, rigid=True, rotation=False, warn_angle_deg=1, min_match_count=10,
                  flann_index_kdtree = 0, flann_trees=5, flann_checks=50):
    """
    co-registers two images and returns the moving image warped to fit target_img and the respective transform matrix
    Script is very close to OpenCV2 image co-registration tutorial:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    only addition/change here: rigid transform & no rotation option
    :param moving_img: image supposed to move
    :param target_img: reference / target image
    :param rigid: if true only tranlation, rotation, and uniform scale
    :param rotation: if false no rotation
    :param warn_angle_deg: cuttoff for warning check if supposed rotation angle bigger in case of rotation=False
    :param min_match_count: min good feature matches
    :param flann_index_kdtree: define algorithm for Fast Library for Approximate Nearest Neighbors - see FLANN doc
    :return: moved/transformed image in target image "space" & transformation matrix
    """
    if len(target_img.shape) > 2:
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    if len(moving_img.shape) > 2:
        moving_img_gray = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)
    else:
        moving_img_gray = moving_img
    height, width = target_img.shape

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(moving_img_gray, None)
    kp2, des2 = sift.detectAndCompute(target_img, None)


    index_params = dict(algorithm=flann_index_kdtree, trees=flann_trees)
    search_params = dict(checks=flann_checks)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if rigid:
            transformation_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                            ransacReprojThreshold=5.0)
            transformation_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
            if not rotation:
                angle = np.arcsin(transformation_matrix[0, 1])
                print('Current rotation {} degrees'.format(np.rad2deg(angle)))
                if abs(np.rad2deg(angle)) > warn_angle_deg:
                    print('Warning: calculated rotation > {} degrees!'.format(warn_angle_deg))
                pure_scale = transformation_matrix[0, 0] / np.cos(angle)
                transformation_matrix[0, 0] = pure_scale
                transformation_matrix[0, 1] = 0
                transformation_matrix[1, 0] = 0
                transformation_matrix[1, 1] = pure_scale
        else:
            transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        transformed_img = cv2.warpPerspective(moving_img, transformation_matrix, (width, height))
        print('Transformation matrix: {}'.format(transformation_matrix))
    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        matchesMask = None
        transformed_img, transformation_matrix = None, None
    return transformed_img, transformation_matrix


def thread_worker_img_processing(payload, output_folder) -> CoRegistrationData:
    i, max_i, moving_img_path, target_img_path = payload
    print('Starting with image {} of {}'.format(i + 1, max_i))
    target_img = cv2.imread(target_img_path)
    # ok this is an error here shape returns h, w, c
    if len(target_img.shape) == 2:
        h, w = target_img.shape
    else:
        h, w, c = target_img.shape
    moving_img = cv2.imread(moving_img_path)

    out, m = register_imgs(moving_img, target_img)
    target_img_name = os.path.splitext(os.path.split(target_img_path)[1])[0]
    moving_img_name = os.path.splitext(os.path.split(moving_img_path)[1])[0]
    cv2.imwrite(os.path.join(output_folder, 'process_out_{}.png'.format(target_img_name)), out)
    print('Done with image {} of {}'.format(i + 1, max_i))
    return CoRegistrationData(name=target_img_name,
                              target_w=w, target_h=h,
                              transform_matrix=m,
                              moving_img_name=moving_img_name)


def coregister_qp_exported_images_to_10x_ref_imgs(moving_imgs_path: List[str],
                                                  target_imgs_path: List[str],
                                                  output_folder: str,
                                                  max_threads: int = 16) -> dict:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    n_imgs = len(moving_imgs_path)
    if n_imgs == len(target_imgs_path):
        if n_imgs > max_threads:
            print('need to split into batches')
            batches = [
                list(zip(range(0, n_imgs), [n_imgs for i in range(0, n_imgs)], moving_imgs_path, target_imgs_path))[i:i + max_threads] for
                i in range(0, n_imgs, max_threads)]
        else:
            batches = [list(zip(range(0, n_imgs), [n_imgs for i in range(0, n_imgs)], moving_imgs_path, target_imgs_path))]
        # print(batches)
        returned_datasets = {}
        for i, batch in enumerate(batches):
            print('starting batch {} of {}'.format(i + 1, len(batches)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:

                future_to_payload = {executor.submit(thread_worker_img_processing, payload, output_folder): payload for payload in batch}
                for future in concurrent.futures.as_completed(future_to_payload):
                    payload = future_to_payload[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (payload, exc))
                    else:
                        print(
                            'image {} of {} - {} processed succesfully - appending data'.format(payload[0] + 1, payload[1],
                                                                                                payload[2]))
                        returned_datasets[data.name] = data
            print('finished batch {} of {}'.format(i + 1, len(batches)))
        print('done with all batches')
        return returned_datasets
    else:
        print('Length of qp_exports and ref_imgs not equal')
        print(moving_imgs_path)
        print(target_imgs_path)
        return {}