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


class MaskNameSplitterInterface:
    """
    Interface for splitting the input / path strings into meaningfull mask class names
    Implement function get_mask_name and pass class in as parameter mask_name_splitter into QuPathBinMaskImporter
    Look at DefaultMaskNameSplitter class for an example
    """
    @staticmethod
    def get_mask_name(input_str: str) -> str:
        pass

class DefaultMaskNameSplitter(MaskNameSplitterInterface):
    @staticmethod
    def get_mask_name(input_str: str) -> str:
        if input_str.split('_')[-3] == "Annotation":
            return input_str.split('_')[-2]
        else:
            return '_'.join([input_str.split('_')[-2], input_str.split('_')[-3]])


class QuPathBinMaskImporter(QuPathBaseImporter):
    _mask_type: str
    img: np.ndarray
    output_file_path: str

    def __init__(self,
                 qp_export_file_path: str,
                 output_folder: str,
                 co_registration_data: CoRegistrationData = None,
                 mask_name_splitter: MaskNameSplitterInterface = None,
                 name: str = None):
        super().__init__(qp_export_file_path=qp_export_file_path,
                         output_folder=output_folder,
                         co_registration_data=co_registration_data,
                         name=name)
        if mask_name_splitter is None:
            self.mask_name_splitter = DefaultMaskNameSplitter
        elif issubclass(mask_name_splitter, MaskNameSplitterInterface):
            self.mask_name_splitter = mask_name_splitter
        else:
            raise ValueError("mask_name_splitter has to be a subclass of MaskNameSplitterInterface")

        self._mask_type = None
        self.img = None
        self.output_file_path = None

    @property
    def mask(self) -> BinaryMask:
        if self.mask_type is not None:
            return BinaryMask(
                name=self.mask_type,
                path=self.output_file_path,
                img=self.img
            )
        else:
            return BinaryMask()

    def _child_check(self) -> bool:
        return True

    def _child_run(self) -> bool:
        moving_img = cv2.imread(self.qp_export_file_path, cv2.IMREAD_GRAYSCALE)
        if self.co_registration_data is not None:
            transformed_img = cv2.warpPerspective(moving_img, self.co_registration_data.transform_matrix, (self.co_registration_data.target_w, self.co_registration_data.target_h))
        else:
            transformed_img = moving_img
        out_file_name = os.path.join(self.output_folder, 'process_out_{}_mask_{}.png'.format(self.name, self.mask_type))
        cv2.imwrite(out_file_name, transformed_img)
        self.img = transformed_img
        self.output_file_path = out_file_name
        return True

    @property
    def mask_type(self) -> str:
        if self._mask_type is None and self.qp_export_file_path is not None:
            self._mask_type = self.mask_name_splitter.get_mask_name(self.qp_export_file_path)
        return self._mask_type

    @staticmethod
    def batch_import(qp_export_path_list: List[str],
                     output_folder: str, mask_name_splitter: MaskNameSplitterInterface = None,
                     co_registration_data_list: List[CoRegistrationData] = None,
                     names: List[str] = None) -> List['QuPathBinMaskImporter']:
        qp_mask_imps = []
        if co_registration_data_list is None:
            co_registration_data_list = [None for i in range(0, len(qp_export_path_list))]
        if names is None:
            names = [None for i in range(0, len(qp_export_path_list))]
        for qp, co_reg_data, name in zip(qp_export_path_list, co_registration_data_list, names):
            qp_mask_imps.append(QuPathBinMaskImporter(qp_export_file_path=qp,
                                                      output_folder=output_folder,
                                                      co_registration_data=co_reg_data,
                                                      mask_name_splitter=mask_name_splitter,
                                                      name=name))
            qp_mask_imps[-1].run()
        return qp_mask_imps






