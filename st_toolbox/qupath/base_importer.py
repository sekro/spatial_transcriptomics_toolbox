"""
QuPathBaseImporter class
Import binary masks form QuPath - via export script
!highly experimental - pre-alpha state
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""

import os
from enum import Enum, unique

# local imports
from .coregistration import CoRegistrationData

@unique
class QuPathExportType(Enum):
    BINARY_MASK = 1
    ANNOTATIONS = 2
    CELLS = 3

class QuPathBaseImporter:
    def __init__(self,
                 qp_export_file_path: str,
                 output_folder: str,
                 co_registration_data: CoRegistrationData):
        self.qp_export_file_path = qp_export_file_path
        self.output_folder = output_folder
        self.co_registration_data = co_registration_data
        self.name = co_registration_data.name

    def check(self) -> bool:
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(self.qp_export_file_path):
            return False
        else:
            if not os.path.isfile(self.qp_export_file_path):
                return False
        if not isinstance(self.co_registration_data, CoRegistrationData):
            return False
        if not self._child_check():
            return False
        return True

    def _child_check(self) -> bool:
        # overwrite in child classes
        return True

    def run(self) -> bool:
        if self.check():
            return self._child_run()
        else:
            # do nothing
            return False

    def _child_run(self) -> bool:
        # overwrite in child classes
        return True

