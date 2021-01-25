"""
QuPathBinMaskImporter class
Import binary masks form QuPath - via export script
!highly experimental - pre-alpha state
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""
from typing import List

import cv2
import numpy as np
import os
from dataclasses import dataclass

# local imports

from .base_importer import QuPathBaseImporter
from .coregistration import CoRegistrationData
from st_toolbox import BinaryMask


@dataclass
class MaskNameSplitter:
    split_string: str
    mask_type_index: int


class QuPathBinMaskImporter(QuPathBaseImporter):
    _mask_type: str
    img: np.ndarray
    output_file_path: str

    def __init__(self,
                 qp_export_file_path: str,
                 output_folder: str,
                 co_registration_data: CoRegistrationData,
                 mask_name_splitter: MaskNameSplitter = None):
        super().__init__(qp_export_file_path=qp_export_file_path,
                         output_folder=output_folder,
                         co_registration_data=co_registration_data)
        if mask_name_splitter is None:
            self.mask_name_splitter = MaskNameSplitter(
                split_string='_',
                mask_type_index=-2
            )
        else:
            self.mask_name_splitter = mask_name_splitter

        self._mask_type = None
        self.img = None
        self.output_file_path = None

    @property
    def mask(self) -> BinaryMask:
        if self._mask_type is not None:
            return BinaryMask(
                name=self._mask_type,
                path=self.output_file_path,
                img=self.img
            )
        else:
            return BinaryMask()

    def _child_check(self) -> bool:
        return True

    def _child_run(self) -> bool:
        moving_img = cv2.imread(self.qp_export_file_path, cv2.IMREAD_GRAYSCALE)
        transformed_img = cv2.warpPerspective(moving_img, self.co_registration_data.transform_matrix, (self.co_registration_data.target_w, self.co_registration_data.target_h))
        out_file_name = os.path.join(self.output_folder, 'process_out_{}_mask_{}.png'.format(self.co_registration_data.name, self.mask_type))
        cv2.imwrite(out_file_name, transformed_img)
        self.img = transformed_img
        self.output_file_path = out_file_name
        return True

    @property
    def mask_type(self) -> str:
        if self._mask_type is None:
            self._mask_type = self.qp_export_file_path.split(self.mask_name_splitter.split_string)[self.mask_name_splitter.mask_type_index]
        return self._mask_type

    @staticmethod
    def batch_import(qp_export_path_list: List[str], co_registration_data_list: List[CoRegistrationData],
                           output_folder: str, mask_name_splitter: MaskNameSplitter = None) -> List['QuPathBinMaskImporter']:
        qp_mask_imps = []
        for qp, co_reg_data in zip(qp_export_path_list, co_registration_data_list):
            qp_mask_imps.append(QuPathBinMaskImporter(qp_export_file_path=qp,
                                                      output_folder=output_folder,
                                                      co_registration_data=co_reg_data,
                                                      mask_name_splitter=mask_name_splitter))
            qp_mask_imps[-1].run()
        return qp_mask_imps






