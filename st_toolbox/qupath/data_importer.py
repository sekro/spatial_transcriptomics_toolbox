"""
QuPathDataImporter class
Import data form QuPath cell detection and or annotations
generated via:
https://github.com/sekro/qupath_scripts/blob/main/watershed_cell_detection.groovy
exported via:
https://github.com/sekro/qupath_scripts/blob/main/export_cells_to_binary_and_json.groovy
from
"""

import json
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum, unique

# local imports
from .base_importer import QuPathBaseImporter
from .coregistration import CoRegistrationData
from .utils import fast_TRS_2d

logger = logging.getLogger(__name__)

@unique
class ObjectTypes(Enum):
    AREA = 1


@dataclass
class Polygon:
    # usage of ndarrays over own 2D points is not very restrictive but more convenient for later use
    centroid: np.ndarray
    contour: np.ndarray


@dataclass
class CellObject:
    name: str
    nucleus: Polygon
    cell: Polygon
    type: ObjectTypes


@dataclass
class AnnotationObject:
    name: str
    shape: Polygon
    type: ObjectTypes


@dataclass
class QuPathDataObject:
    name: str
    img_file_base_name: str
    downsample: int
    org_px_height_micron: float
    org_px_width_micron: float
    org_px_avg_micron: float
    cells: [CellObject]
    annotations: [AnnotationObject]


class QuPathDataImporter(QuPathBaseImporter):
    _qp_json: dict
    img: np.ndarray
    output_path: str
    data: QuPathDataObject

    def __init__(self,
                 qp_export_file_path: str,
                 output_folder: str,
                 co_registration_data: CoRegistrationData = None,
                 name: str = None):
        super().__init__(qp_export_file_path=qp_export_file_path,
                         output_folder=output_folder,
                         co_registration_data=co_registration_data,
                         name=name)
        self._qp_json = None
        self.data = None
        self.default_type = ObjectTypes.AREA
        self.qp_type_map = {
            "AREA": ObjectTypes.AREA
        }

    def _child_check(self) -> bool:
        if self._qp_json is None:
            return self._read_json_data_file()
        return True

    def _child_run(self) -> bool:
        return self._decode_json()

    def _read_json_data_file(self):
        with open(self.qp_export_file_path, "r") as read_file:
            self._qp_json = json.load(read_file)
        return True

    def _decode_json(self):
        self.data = QuPathDataObject(
            name=self._qp_json["name"],
            img_file_base_name=self._qp_json["imgFileBaseName"],
            downsample=int(self._qp_json["downsample"]),
            org_px_height_micron=float(self._qp_json["pixelHeightMicron"]),
            org_px_width_micron=float(self._qp_json["pixelWidthMicron"]),
            org_px_avg_micron=float(self._qp_json["avgPixelSizeMicron"]),
            cells=[],
            annotations=[]
        )
        if "annotations" in self._qp_json:
            for qp_annotation in self._qp_json["annotations"]:
                self.data.annotations.append(self.qp_annotation2annotation(qp_annotation))
        if "cells" in self._qp_json:
            for qp_cell in self._qp_json["cells"]:
                self.data.cells.append(self.qp_cell2cell(qp_cell=qp_cell))
        return True

    def qp_centroid2centroid(self, qp_centroid_x, qp_centroid_y):
        if self.co_registration_data is None:
            return np.divide(np.array([np.float64(qp_centroid_x),
                                               np.float64(qp_centroid_y)]),
                                     self.data.downsample)
        else:
            return fast_TRS_2d(np.divide(np.array([np.float64(qp_centroid_x),
                                                   np.float64(qp_centroid_y)]),
                                         self.data.downsample),
                               transform_matrix=self.co_registration_data.transform_matrix,
                               input_is_point=True)

    def qp_poly2poly(self, qp_poly):
        poly = np.zeros((0, 2), dtype=np.float64)
        for point in qp_poly:
            poly = np.append(poly, [[np.float64(point["x"]), np.float64(point["y"])]], axis=0)
        if self.co_registration_data is None:
            logger.debug("no coreg-data present - downsampling of poly only")
            return np.divide(poly, self.data.downsample)
        else:
            logger.debug("coregistration and downsampling of poly")
            return fast_TRS_2d(np.divide(poly, self.data.downsample), transform_matrix=self.co_registration_data.transform_matrix)

    def qp_annotation2annotation(self, qp_annotation, cstr: str = None):
        if cstr is None:
            common_str = 'Annotation'
        else:
            common_str = cstr
        if qp_annotation["name"] == common_str:
            name = qp_annotation["className"]
        elif common_str in qp_annotation["name"]:
            name = qp_annotation["className"] + qp_annotation["name"].replace(common_str,'')
        else:
            name = qp_annotation["className"] + qp_annotation["name"]
        annotation = AnnotationObject(
            name=name,
            type=self.qp_type2type(qp_annotation["type"]),
            shape=Polygon(
                centroid=self.qp_centroid2centroid(qp_annotation["shape"]["centroidx"],
                                                   qp_annotation["shape"]["centroidy"]),
                contour=self.qp_poly2poly(qp_poly=qp_annotation["shape"]["polygon"])
            )
        )
        return annotation

    def qp_cell2cell(self, qp_cell):
        cell = CellObject(
            name=qp_cell["name"],
            type=self.qp_type2type(qp_cell["type"]),
            cell=Polygon(
                centroid=self.qp_centroid2centroid(qp_cell["cell_poly"]["centroidx"],
                                                   qp_cell["cell_poly"]["centroidy"]),
                contour=self.qp_poly2poly(qp_poly=qp_cell["cell_poly"]["polygon"])
            ),
            nucleus=Polygon(
                centroid=self.qp_centroid2centroid(qp_cell["nucleus_poly"]["centroidx"],
                                                   qp_cell["nucleus_poly"]["centroidy"]),
                contour=self.qp_poly2poly(qp_poly=qp_cell["nucleus_poly"]["polygon"])
            )
        )
        return cell

    def qp_type2type(self, qp_type):
        if qp_type in self.qp_type_map:
            return self.qp_type_map[qp_type]
        else:
            return self.default_type

