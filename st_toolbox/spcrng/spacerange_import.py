"""
SpaceRangerImporter Class for import of reads from h5 files and spatial info as obtained from 10x spaceranger pipeline
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""

import json
import os
import logging
import pickle
from dataclasses import dataclass
from enum import Enum, unique
from typing import List

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import tables

from st_toolbox import BinaryMask
from st_toolbox.qupath import QuPathDataObject

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)

@unique
class SpaceRangerDataType(Enum):
    RAW = 1
    FILTERED = 2


@dataclass
class SpaceRangerScaleFactors:
    spot_diameter_fullres: float = 0.
    tissue_hires_scalef: float = 0.
    fiducial_diameter_fullres: float = 0.
    tissue_lowres_scalef: float = 0.


@dataclass
class SpaceRangerCountMatrix:
    feature_ref: dict = None
    barcodes: list = None
    matrix: sp_sparse.csc_matrix = None


@dataclass
class SpaceRangerPaths:
    name: str = None
    tissue_hires_img: str = None
    filtered_matrix_h5: str = None
    raw_matrix_h5: str = None
    scalefactors_json: str = None
    tissue_pos_csv: str = None


@dataclass
class SpaceRangerRun:
    name: str
    paths: SpaceRangerPaths
    spcrng_matrix: SpaceRangerCountMatrix = None
    data_type: SpaceRangerDataType = None
    df: pd.DataFrame = None
    scale_factors: SpaceRangerScaleFactors = None


@dataclass
class DataFrameColumnNames:
    BARCODE: str = "barcode"
    SPCRNG_TISSUE: str = "tissue_detection_spaceranger"
    SPOT_ROW: str = "row"
    SPOT_COL: str = "col"
    SPOT_IMG_ROW: str = "image_row"
    SPOT_IMG_COL: str = "image_col"
    TOTAL_UMI: str = "total_UMI_reads"


@dataclass
class SpaceRangerSpot:
    barcode: str
    slide_id: str
    img_row: float
    img_col: float
    img_scalefactor: float
    diameter_px: float
    masks: List[BinaryMask] = None
    metadata: pd.Series = None
    reads: pd.Series = None
    n_cells: int = 0

    @property
    def unique_name(self):
        return '_'.join([self.slide_id, self.barcode])


@dataclass
class SpaceRangerSpots:
    _spots: List[SpaceRangerSpot]
    known_barcodes: List[str]
    img: np.ndarray
    _df: pd.DataFrame
    _attached_qp_data: QuPathDataObject

    def __init__(self, barcodes: List[str]=[], slide_id: List[str]=[],
                 img_rows: List[float]=[], img_cols: List[float]=[], img_scalefactors: List[float]=None,
                 diameter_px=None, masks: List[List[BinaryMask]]=None,
                 reads: List[pd.Series]=None,
                 img: np.ndarray=None):
        self._spots = []
        self.img = img
        self._df = None
        self._attached_qp_data = None
        self.force_regen_df = True
        if len(barcodes) > 0:
            if len(barcodes) == len(img_rows) and len(barcodes) == len(img_cols):
                if reads is not None:
                    if not (isinstance(reads, list) and len(reads) == len(barcodes)):
                        raise ValueError("reads has to be a list of pandas.Series or None")
                    if not all([isinstance(item, pd.Series) for item in reads]):
                        raise ValueError("reads has to be a list of pandas.Series or None")
                else:
                    reads = [None for i in range(0, len(barcodes))]
                if masks is not None:
                    if not (isinstance(masks, list) and len(masks) == len(barcodes)):
                        raise ValueError("masks has to be a list (same length as barcodes) of list of BinaryMask or None")
                    if not all([isinstance(item, list) for item in masks]):
                        raise ValueError("masks has to be a list (same length as barcodes) of list of BinaryMask or None")
                    for list_of_binmasks in masks:
                        if not all([isinstance(item, BinaryMask) for item in list_of_binmasks]):
                            raise ValueError(
                                "masks has to be a list (same length as barcodes) of list of BinaryMask or None")
                else:
                    masks = [[] for i in range(0, len(barcodes))]
                if isinstance(slide_id, list):
                    if len(slide_id) == len(barcodes):
                        s_ids = slide_id
                    elif len(slide_id) == 0:
                        s_ids = ['unknown' for i in range(0, len(barcodes))]
                    else:
                        raise ValueError("slide_id has to be either a list of strings (same length as barcodes) or a single string")
                elif isinstance(slide_id, str):
                    s_ids = [slide_id for i in range(0, len(barcodes))]
                else:
                    raise ValueError("slide_id has to be either a list of strings (same length as barcodes) or a single string")
                if diameter_px is None:
                    diameters = [0. for i in range(0, len(barcodes))]
                else:
                    if isinstance(diameter_px, list) and len(diameter_px) == len(barcodes):
                        diameters = diameter_px
                    elif isinstance(diameter_px, (float, int)):
                        diameters = [diameter_px for i in range(0, len(barcodes))]
                    else:
                        raise ValueError("diameter_px has to be either a list of diameters (same length as barcodes) or a single diameter of type float or int")
                if img_scalefactors is None:
                    img_scalefactors = [1. for i in range(0, len(barcodes))]
                else:
                    if isinstance(img_scalefactors, list) and len(img_scalefactors) == len(barcodes):
                        if not all([isinstance(item, (float, int)) for item in img_scalefactors]):
                            raise ValueError(
                                "img_scalefactors has to be either a list of scalefactors (same length as barcodes) or a single scalefactor of type float or int")
                        i_sf = img_scalefactors
                    elif isinstance(img_scalefactors, (float, int)):
                        i_sf = [img_scalefactors for i in range(0, len(barcodes))]
                    else:
                        raise ValueError(
                            "img_scalefactors has to be either a list of scalefactors (same length as barcodes) or a single scalefactor of type float or int")
                for barcode, s_id, i_row, i_col, i_scf, d, bin_masks, r in zip(barcodes, s_ids,
                                                                        img_rows, img_cols, i_sf,
                                                                        diameters, masks, reads):
                    self._spots.append(
                        SpaceRangerSpot(
                            barcode=barcode,
                            slide_id=s_id,
                            img_row=i_row,
                            img_col=i_col,
                            img_scalefactor=i_scf,
                            diameter_px=d,
                            masks=bin_masks,
                            reads=r
                        )
                    )
                self.known_barcodes = [s.barcode for s in self._spots]
            else:
                raise ValueError("all lists must have same length")


    def __len__(self):
        return len(self._spots)

    def __getitem__(self, item):
        return self._spots[item]

    def __setitem__(self, key, value):
        if isinstance(value, SpaceRangerSpot):
            self._spots[key] = value
            self.known_barcodes = [s.barcode for s in self._spots]
        else:
            raise ValueError("provided value has to be of type SpaceRangerSpot")

    @property
    def name(self):
        if len(self._spots) > 0 and isinstance(self._spots[0], SpaceRangerSpot):
            return self._spots[0].slide_id
        else:
            return "unnamed"

    @property
    def qp_data(self) -> QuPathDataObject:
        if self._attached_qp_data is not None:
            return self._attached_qp_data
        else:
            return QuPathDataObject(
                name='',
                img_file_base_name='',
                downsample=0,
                org_px_height_micron=0,
                org_px_width_micron=0,
                org_px_avg_micron=0,
                cells=[],
                annotations=[]
            )

    @qp_data.setter
    def qp_data(self, data: QuPathDataObject):
        if isinstance(data, QuPathDataObject):
            self._attached_qp_data = data
        else:
            logger.warning("Input has to be of type QuPathDataObject")

    @property
    def df(self) -> pd.DataFrame:
        if self.force_regen_df:
            logger.debug('start gen of df')
            df = pd.DataFrame()
            series = []
            for spot in self._spots:
                series.append(
                    pd.Series(
                        data=[spot.barcode, spot.img_col, spot.img_row, spot.n_cells],
                        index=['barcode', 'img_col', 'img_row', 'n_cells'],
                        name=spot.unique_name
                    ).append(
                        [
                            spot.metadata.rename(spot.unique_name),
                            spot.reads.rename(spot.unique_name)
                        ]
                    )
                )
            self.force_regen_df = False
            self._df = df.append(series, verify_integrity=True)
            logger.debug('done gen of df')
        return self._df

    def save_to_disk(self, location: str):
        if len(self._spots) > 0:
            if not os.path.exists(location):
                os.mkdir(path=location)
            if not os.path.isfile(location):
                filename = "{}.feather".format(self._spots[0].slide_id)
                self.df.to_feather(path=os.path.join(location, filename))
                logger.info("Dataframe saved to {}".format(os.path.join(location, filename)))
            else:
                logger.warning("File exits - saving nothing")

    def index(self, barcode):
        if isinstance(barcode, str):
            if barcode in self.known_barcodes:
                return self.known_barcodes.index(barcode)
            else:
                return None
        else:
            raise ValueError("barcode needs to be a string")

    def append(self, item):
        if isinstance(item, SpaceRangerSpot):
            self._spots.append(item)
            self.known_barcodes.append(item.barcode)
            self.force_regen_df = True
        else:
            raise ValueError("value has to be of type SpaceRangerSpot")

    def remove(self, barcode):
        if isinstance(barcode, str):
            if barcode in self.known_barcodes:
                del self._spots[self.known_barcodes.index(barcode)]
                self.known_barcodes.remove(barcode)
                self.force_regen_df = True
            else:
                raise ValueError('unknown barcode')
        else:
            raise ValueError('barcode has to be a string')

    def items(self):
        for index, spot in enumerate(self._spots):
            yield index, spot

    def spots(self):
        for spot in self._spots:
            yield spot

    def coordinates(self):
        for spot in self._spots:
            yield spot.barcode,  spot.img_row, spot.img_col

    def masks(self):
        for spot in self._spots:
            yield spot.barcode, spot.masks

    def reads(self):
        for spot in self._spots:
            yield spot.barcode, spot.reads


class SpaceRangerImporter:
    def __init__(self, paths: SpaceRangerPaths, df_col_names: DataFrameColumnNames = DataFrameColumnNames(),
                 use_hires_img: bool = True):
        self._data = SpaceRangerRun(name=paths.name,
                                    paths=paths)
        self._df_col_names = df_col_names
        self._df_reads_col_names: pd.Index = None
        self._use_highres_img: bool = use_hires_img
        self._tissue_pos_csv_col_names = [self._df_col_names.BARCODE,
                                          self._df_col_names.SPCRNG_TISSUE,
                                          self._df_col_names.SPOT_ROW,
                                          self._df_col_names.SPOT_COL,
                                          self._df_col_names.SPOT_IMG_ROW,
                                          self._df_col_names.SPOT_IMG_COL]

    def _check(self):
        # TODO implement file exists checks
        return True

    @property
    def df(self) -> pd.DataFrame:
        return self._data.df

    @property
    def data(self) -> SpaceRangerRun:
        return self._data

    @property
    def spots(self) -> SpaceRangerSpots:
        if self._data.df is not None:
            if self._use_highres_img:
                diameter = self._data.scale_factors.spot_diameter_fullres
                sf = 1
            else:
                diameter = self._data.scale_factors.spot_diameter_fullres * self._data.scale_factors.tissue_hires_scalef
                sf = self._data.scale_factors.tissue_hires_scalef
            return SpaceRangerSpots(barcodes=self._data.df.index.to_list(),
                                    slide_id=self._data.name,
                                    img_rows=self._data.df[self._df_col_names.SPOT_IMG_ROW].to_list(),
                                    img_cols=self._data.df[self._df_col_names.SPOT_IMG_COL].to_list(),
                                    diameter_px=diameter,
                                    img_scalefactors=sf,
                                    reads=[row[1] for row in self._data.df[self._df_reads_col_names].iterrows()],
                                    img=cv2.cvtColor(src=cv2.imread(self._data.paths.tissue_hires_img,
                                                                    cv2.IMREAD_COLOR),
                                                     code=cv2.COLOR_BGR2RGB))
        else:
            return SpaceRangerSpots()

    def load_data(self, filtered_data: bool = False) -> pd.DataFrame:
        if self._check():
            if filtered_data:
                self._data.spcrng_matrix = self.get_matrix_from_h5(self._data.paths.filtered_matrix_h5)
                self._data.data_type = SpaceRangerDataType.FILTERED
            else:
                self._data.spcrng_matrix = self.get_matrix_from_h5(self._data.paths.raw_matrix_h5)
                self._data.data_type = SpaceRangerDataType.RAW
            self._load_scalefactors()
            _reads = self._get_reads_df()
            self._data.df = pd.concat([self._get_tissue_pos_df(rows=_reads.index.to_list()), _reads], axis=1)
            return self._data.df

    def _load_scalefactors(self):
        with open(self._data.paths.scalefactors_json, 'r') as json_file:
            tmp_dict = json.load(json_file)
            self._data.scale_factors = SpaceRangerScaleFactors(
                spot_diameter_fullres=float(tmp_dict["spot_diameter_fullres"]),
                tissue_hires_scalef=float(tmp_dict["tissue_hires_scalef"]),
                fiducial_diameter_fullres=float(tmp_dict["fiducial_diameter_fullres"]),
                tissue_lowres_scalef=float(tmp_dict["tissue_lowres_scalef"])
            )

    def _get_tissue_pos_df(self, rows: list=None) -> pd.DataFrame:
        _df = pd.read_csv(self._data.paths.tissue_pos_csv, header=None)
        _df.columns = self._tissue_pos_csv_col_names
        _df = _df.sort_values(self._df_col_names.BARCODE)
        _df = _df.set_index(self._df_col_names.BARCODE)
        # tissue_hires_img is the unscaled high res input into spaceranger
        # spaceranger downscales these to largest dim 2000 px max
        # scalefactor transforms between those
        # spot img row & col values from csv are in tissue_hires_img scale!
        if not self._use_highres_img:
            _df[self._df_col_names.SPOT_IMG_ROW] = _df[self._df_col_names.SPOT_IMG_ROW] * self._data.scale_factors.tissue_hires_scalef
            _df[self._df_col_names.SPOT_IMG_COL] = _df[self._df_col_names.SPOT_IMG_COL] * self._data.scale_factors.tissue_hires_scalef
        if rows is None:
            return _df
        else:
            return _df.loc[rows, :]

    def _get_reads_df(self) -> pd.DataFrame:
        col_names = []
        for fid, fname in zip(self._data.spcrng_matrix.feature_ref['id'],
                              self._data.spcrng_matrix.feature_ref['name']):
            col_names.append(fname + " - " + fid)
        _df = pd.DataFrame(data=self._data.spcrng_matrix.matrix.T.toarray(), columns=col_names,
                           index=self._data.spcrng_matrix.barcodes)
        _df[self._df_col_names.TOTAL_UMI] = _df.sum(axis=1)
        self._df_reads_col_names = _df.columns
        return _df

    @staticmethod
    def pickle_spots(spots: SpaceRangerSpot, location: str):
        if len(spots) > 0:
            if not os.path.exists(location):
                os.mkdir(path=location)
            if not os.path.isfile(location):
                filename = "{}.pickle".format(spots[0].slide_id)
                with open(os.path.join(location, filename), 'wb') as f:
                    pickle.dump(spots, f, pickle.HIGHEST_PROTOCOL)
                logger.info("Spots saved to {}".format(os.path.join(location, filename)))
            else:
                logger.warning("File exits - saving nothing")

    @staticmethod
    def unpickle_spots(path_to_pickle: str) -> SpaceRangerSpots:
        if os.path.isfile(path_to_pickle):
            logger.warning("Only unpickle from trusted source!!!")
            with open(path_to_pickle, 'rb') as f:
                spots = pickle.load(f)
            if isinstance(spots, SpaceRangerSpots):
                logger.info("spots loaded from pickle")
                return spots
            else:
                logger.info("unpickled object/data is not of type SpaceRangerSpots")
                return SpaceRangerSpots()
        else:
            logger.warning("File not found")

    @staticmethod
    def get_matrix_from_h5(h5_file_path: str) -> SpaceRangerCountMatrix:
        """
        Read a spacerange h5 file - adapted from 10x example
        :param h5_file_path:
        :return:
        """
        if os.path.isfile(h5_file_path):
            with tables.open_file(h5_file_path, 'r') as f:
                mat_group = f.get_node(f.root, 'matrix')
                barcodes = f.get_node(mat_group, 'barcodes').read().astype(str)
                data = getattr(mat_group, 'data').read()
                indices = getattr(mat_group, 'indices').read()
                indptr = getattr(mat_group, 'indptr').read()
                shape = getattr(mat_group, 'shape').read()
                matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

                feature_ref = {}
                feature_group = f.get_node(mat_group, 'features')
                feature_ids = getattr(feature_group, 'id').read()
                feature_names = getattr(feature_group, 'name').read()
                feature_types = getattr(feature_group, 'feature_type').read()
                feature_ref['id'] = feature_ids.astype(str)
                feature_ref['name'] = feature_names.astype(str)
                feature_ref['feature_type'] = feature_types.astype(str)
                tag_keys = getattr(feature_group, '_all_tag_keys').read().astype(str)
                # print('Tag keys: {}'.format(tag_keys))
                for key in tag_keys:
                    # key = key.decode('UTF-8')
                    feature_ref[key] = getattr(feature_group, key).read()
            return SpaceRangerCountMatrix(feature_ref, barcodes, matrix)
        else:
            return SpaceRangerCountMatrix()
