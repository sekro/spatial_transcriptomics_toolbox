"""
Utility methods - part of ST toolbox
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""


from typing import Tuple, List

import pandas as pd
import cv2

from .spacerange_import import SpaceRangerSpots


def inside_circle(radius: float, circle_center: Tuple[float, float], point: Tuple[float, float]) -> bool:
    return (point[0] - circle_center[0]) ** 2 + (point[1] - circle_center[1]) ** 2 <= radius ** 2


def export_to_one_dataframe(spots_sets: List[SpaceRangerSpots]) -> pd.DataFrame:
    df = pd.DataFrame()
    for spots in spots_sets:
        for spot in spots:
            unique_name = '_'.join([spot.slide_id, spot.barcode])
            df = df.append(
                pd.Series(
                    data=[spot.barcode, spot.img_col, spot.img_row],
                    index=['barcode', 'img_col', 'img_row'],
                    name=unique_name
                ).append(
                    [
                        spot.metadata.rename(unique_name),
                        spot.reads.rename(unique_name)
                    ]
                ),
                verify_integrity=True
            )
    return df