"""
class HistoPathMerge - merge histopathology data with spatial transcriptomics data
part of st_toolbox
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""
from typing import Tuple, List
from dataclasses import dataclass
import logging
import cv2
import numpy as np
import pandas as pd

from st_toolbox import BinaryMask
from .spacerange_import import SpaceRangerSpots, SpaceRangerSpot
from .utils import inside_circle
from ..qupath.data_importer import QuPathDataObject, CellObject

GLOBAL_debug = True
logger = logging.getLogger(__name__)

if GLOBAL_debug:
    logging.basicConfig(level=logging.WARN)


@dataclass
class ExclusiveMaskPair:
    mask_1: BinaryMask
    mask_2: BinaryMask


class HistoPathMerger:
    def __init__(self, masks: List[BinaryMask], spots: SpaceRangerSpots,
                 exclusive_mask_pairs: List[ExclusiveMaskPair] = None,
                 qp_data: QuPathDataObject = None):
        self._exclusive_mask_pairs = exclusive_mask_pairs
        self.masks = masks
        self.spots = spots
        self.qp_data: QuPathDataObject = qp_data
        self._canny_edge_thr = (100, 200)

    def merge(self, auto_gen_border_masks: bool = True, border_only_outside: bool = False,
              border_width: int = 100, auto_scale: bool = True) -> pd.DataFrame:
        # TODO: border size as µm and or "spots"
        if not border_only_outside:
            border_width = int(round(border_width / 2))
        if auto_scale:
            # in this case we have to upscale the border so that the final border has the requested width
            logger.debug("autoscaling")
            _border_width = int(round(border_width / HistoPathMerger.get_scale_factor(spots=self.spots)))
        else:
            _border_width = border_width
        kernel = np.ones((_border_width, _border_width), np.uint8)
        logger.debug("Kernel: {}, _border_width: {}, scalefactor: {}".format(kernel, _border_width, HistoPathMerger.get_scale_factor(spots=self.spots)))
        if self._exclusive_mask_pairs is not None:
            logger.info("start pre-processing exclusive mask pairs")
            for pair in self._exclusive_mask_pairs:
                self.masks.extend(HistoPathMerger.pre_process_exclusive_mask_pairs(mask_pair=pair, spots=self.spots))
            logger.info("done pre-processing exclusive mask pairs")
        if auto_gen_border_masks:
            logger.info("start generating mask borders")
            borders = []
            for mask in self.masks:
                #logger.debug("mask {} - start edge detetection".format(mask.name))
                #_edge = cv2.Canny(image=mask.img, threshold1=self._canny_edge_thr[0],
                #                  threshold2=self._canny_edge_thr[1])
                #logger.debug("mask {} - done edge detetection / start dilate".format(mask.name))

                borders.append(HistoPathMerger.generate_border_mask(mask=mask, kernel=kernel,
                                                                    border_only_outside=border_only_outside))
                logger.debug("mask {} done".format(mask.name))
            self.masks.extend(borders)
            logger.info("done generating mask borders")
        logger.info("start processing all binary masks")
        for mask in self.masks:
            self.spots = HistoPathMerger.get_data_from_mask(mask=mask, spots=self.spots, auto_scale=auto_scale)
        logger.info("done processing all binary masks")
        if self.qp_data is not None:
            # TODO - implement annotations as well
            logger.info("processing cell data")
            self.spots = HistoPathMerger.get_cells_per_spot(self.spots, self.qp_data.cells)
            logger.info("done processing cell data")
            logger.info("attaching qp_data to spots obj")
            self.spots.qp_data = self.qp_data
            logger.info("qp_data attached")
        # last action - (re)build full dataframe of spots obj
        self.spots.force_regen_df = True
        return self.spots.df

    @staticmethod
    def generate_border_mask(mask: BinaryMask, kernel: np.ndarray, border_only_outside: bool) -> BinaryMask:
        if border_only_outside:
            _border = (cv2.dilate(mask.img, kernel, iterations=1).astype(np.bool) & np.bitwise_not(
                mask.img.astype(np.bool))).astype(np.uint8)
        else:
            _border = (cv2.dilate(mask.img, kernel, iterations=1).astype(np.bool) & np.bitwise_not(
                cv2.erode(mask.img, kernel, iterations=1).astype(np.bool))).astype(np.uint8)
        logger.debug("mask {} - done dilate / setup binmask".format(mask.name))
        border = BinaryMask(
            name=mask.name + "_autogen_border",
            img=_border * 255
        )
        if GLOBAL_debug:
            logger.debug("Saving autogen border mask to disk")
            cv2.imwrite("autoborder_{}.png".format(mask.name), border.img)
        return border

    @staticmethod
    def get_cells_per_spot(spots: SpaceRangerSpots, cells: List[CellObject], auto_scale=True) -> SpaceRangerSpots:
        if auto_scale:
            scale_factor = HistoPathMerger.get_scale_factor(spots=spots)
        else:
            scale_factor = 1.
        for cell in cells:
            scaled_cell_center = tuple(cell.nucleus.centroid * scale_factor)
            for spot in spots:
                if inside_circle(radius=spot.diameter_px / 2, circle_center=(spot.img_col, spot.img_row),
                                 point=scaled_cell_center):
                    spot.n_cells += 1
                    break
        return spots

    @staticmethod
    def get_scale_factor(spots: SpaceRangerSpots) -> float:
        if all([spots[0].img_scalefactor == s.img_scalefactor for s in spots.spots()]):
            scale_factor = spots[0].img_scalefactor
        else:
            raise Warning(
                'spots seem to belong to different imgs with unequal scale factors - cannot process in one run')
        return scale_factor

    @staticmethod
    def get_scaled_mask(mask: BinaryMask, scale_factor: float) -> BinaryMask:
        if scale_factor != 1:
            if len(mask.img.shape) == 2:
                h, w = mask.img.shape
            else:
                h, w, c = mask.img.shape
            h = int(round(h * scale_factor))
            w = int(round(w * scale_factor))
            tmp_img = cv2.resize(mask.img, (w, h), interpolation=cv2.INTER_AREA)
            logger.debug("mask scaling: mask name {} - org shape {} new shape {}".format(mask.name, mask.img.shape, tmp_img.shape))
        else:
            tmp_img = mask.img
            logger.debug("mask scaling: nothing do to scalefactor == 1")
        return BinaryMask(name=mask.name,
                           img=tmp_img)

    @staticmethod
    def get_exclusive_mask(mask: BinaryMask, overlap: BinaryMask) -> BinaryMask:
        return BinaryMask(
            name='_'.join([mask.name, 'exclusive']),
            img=(mask.img.astype(np.bool) & np.bitwise_not(overlap.img.astype(np.bool))).astype(np.uint8)
        )

    @staticmethod
    def get_masked_spot(spot_center: Tuple[int, int], spot_diameter: int, mask: BinaryMask) -> Tuple[float, BinaryMask]:
        cy, cx = spot_center
        if not spot_diameter % 2:
            box_size = spot_diameter + 1
        else:
            box_size = spot_diameter
        box_grid_coord = np.mgrid[:box_size, :box_size]
        circle_r_squared_grid = (box_grid_coord[0] - int(box_size / 2)) ** 2 + (box_grid_coord[1] - int(box_size / 2)) ** 2
        circle_mask = circle_r_squared_grid <= ((spot_diameter / 2) ** 2)
        half_box_offset = int(box_size / 2)
        img_box = mask.img[cy - half_box_offset: cy + half_box_offset + 1, cx - half_box_offset: cx + half_box_offset + 1].astype(np.bool)

        try:
            masked_spot_fraction = np.sum(img_box & circle_mask ) /np.sum(circle_mask)
        except:
            logger.debug(
                "spot_center: {}, spot diameter: {}, box_size: {}, half_box_offset: {}".format(spot_center, spot_diameter,
                                                                                           box_size, half_box_offset))
            logger.debug("img_box: {} shape {}, circle mask: {} shape {}".format(img_box, img_box.shape, circle_mask,
                                                                                 circle_mask.shape))
            logger.debug("mask name {} - mask shape {}".format(mask.name, mask.img.shape))
        spot_mask = BinaryMask(name=mask.name,
                               img=(img_box & circle_mask).astype(np.uint8))
        return masked_spot_fraction, spot_mask

    @staticmethod
    def pre_process_exclusive_mask_pairs(mask_pair: ExclusiveMaskPair, spots: SpaceRangerSpots=None) -> List[BinaryMask]:
        if spots is not None:
            if HistoPathMerger.is_size_equal_mask_and_img(img=spots.img, mask=mask_pair.mask_1):
                m1 = mask_pair.mask_1
            else:
                m1 = HistoPathMerger.get_scaled_mask(mask=mask_pair.mask_1,
                                                       scale_factor=HistoPathMerger.get_scale_factor(spots=spots))
            if HistoPathMerger.is_size_equal_mask_and_img(img=spots.img, mask=mask_pair.mask_2):
                m2 = mask_pair.mask_2
            else:
                m2 = HistoPathMerger.get_scaled_mask(mask=mask_pair.mask_2,
                                                     scale_factor=HistoPathMerger.get_scale_factor(spots=spots))

            pair = ExclusiveMaskPair(
                mask_1=m1,
                mask_2=m2
            )
        else:
            pair = mask_pair
        _overlap = BinaryMask(
            name='_'.join([pair.mask_1.name, pair.mask_2.name, 'overlap']),
            img=(pair.mask_1.img.astype(np.bool) & pair.mask_2.img.astype(np.bool)).astype(np.uint8)
        )
        return [HistoPathMerger.get_exclusive_mask(pair.mask_1, _overlap),
                HistoPathMerger.get_exclusive_mask(pair.mask_2, _overlap),
                _overlap]

    @staticmethod
    def is_size_equal_mask_and_img(img: np.ndarray, mask: BinaryMask) -> bool:
        if img is not None and mask.img is not None:
            if len(img.shape) == 2:
                h1, w1 = img.shape
            else:
                h1, w1, c = img.shape
            if len(mask.img.shape) == 2:
                h2, w2 = mask.img.shape
            else:
                h2, w2, c = mask.img.shape
            logger.debug("img and mask compare: img w {} h {}, m w {} h {} - return val {}".format(w1, h1, w2, h2, str(h1 == h2 and w1 == w2)))
            return h1 == h2 and w1 == w2
        else:
            return False

    @staticmethod
    def get_data_from_mask(mask: BinaryMask, spots: SpaceRangerSpots, auto_scale: bool=True) -> SpaceRangerSpots:
        if auto_scale and not HistoPathMerger.is_size_equal_mask_and_img(img=spots.img, mask=mask):
            _mask = HistoPathMerger.get_scaled_mask(mask=mask,
                                                    scale_factor=HistoPathMerger.get_scale_factor(spots=spots))
        else:
            _mask = mask
        for spot in spots.spots():
            spot_center = (int(round(spot.img_row)), int(round(spot.img_col)))
            masked_spot_fraction, spot_mask = HistoPathMerger.get_masked_spot(spot_center=spot_center,
                                                                              spot_diameter=int(round(spot.diameter_px)),
                                                                              mask=_mask)
            if spot.masks is None:
                spot.masks = []
            if isinstance(spot.masks, list):
                spot.masks.append(spot_mask)
            else:
                raise Warning("Something is wrong with your spot objects - masks should be a list but is not for {}".format(spot.barcode))
            if spot.metadata is None:
                spot.metadata = pd.Series([], name=spot.barcode, dtype=np.float64)
            if isinstance(spot.metadata, pd.Series):
                spot.metadata[spot_mask.name] = masked_spot_fraction
            else:
                raise Warning("Something is wrong with your spot objects - metadata should be a pandas.Series but is not for {}".format(spot.barcode))
        return spots