"""
Batch processing functions
Part of st_toolbox
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""
from typing import List, Tuple
import pandas as pd
import logging
import os
from dataclasses import dataclass
from st_toolbox.qupath import QuPathBinMaskImporter, QuPathDataImporter, QuPathDataObject, MaskNameSplitterInterface
from st_toolbox.qupath.coregistration import CoRegistrationData
from st_toolbox.spcrng import SpaceRangerPaths, SpaceRangerImporter, SpaceRangerSpots
from st_toolbox.spcrng.merge_data import HistoPathMerger, ExclusiveMaskPair

GLOBAL_debug = False
logger = logging.getLogger(__name__)

if GLOBAL_debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARN)

@dataclass
class BatchPaths:
    id: str
    qp_masks: List[str]
    qp_mask_name_splitter: MaskNameSplitterInterface
    qp_data: str
    spcrng_paths: SpaceRangerPaths
    coreg_data: CoRegistrationData = None
    use_qp_hes: bool = False


def batch_merge(batches: List[BatchPaths], output_folder: str, pickle_spots = False,
                special_mask_names: List[Tuple[str, str]] = None) -> Tuple[List[SpaceRangerSpots],
                                                                           List[QuPathDataObject],
                                                                           List[pd.DataFrame]]:

    df_collector = []
    qpd_collector = []
    spots_collector = []
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if special_mask_names is not None:
        _special_filter_lst = []
        for pair in special_mask_names:
            _special_filter_lst.extend(list(pair))
    for bp in batches:
        logger.info("Starting with {}".format(bp.id))
        _qpms = QuPathBinMaskImporter.batch_import(qp_export_path_list=bp.qp_masks,
                                                   mask_name_splitter=bp.qp_mask_name_splitter,
                                                   output_folder=os.path.join(output_folder, 'qp_mask_import'),
                                                   co_registration_data_list=[bp.coreg_data for i in range(0, len(bp.qp_masks))],
                                                   names=[bp.id for i in range(0, len(bp.qp_masks))])
        if special_mask_names is None:
            _masks = [q.mask for q in _qpms]
            _excl_mask_pairs = None
        else:
            _masks = [q.mask for q in _qpms if q.mask.name not in _special_filter_lst]
            _specials = [q.mask for q in _qpms if q.mask.name in _special_filter_lst]
            _spec_dic = {}
            for m in _specials:
                _spec_dic[m.name] = m
            _excl_mask_pairs = []
            for pair in special_mask_names:
                _excl_mask_pairs.append(ExclusiveMaskPair(
                    mask_1=_spec_dic[pair[0]],
                    mask_2=_spec_dic[pair[1]],
                ))
        _qpd = QuPathDataImporter(qp_export_file_path=bp.qp_data,
                                  output_folder=os.path.join(output_folder, 'qp_data_import'),
                                  co_registration_data=bp.coreg_data,
                                  name=bp.id)
        _qpd.run()
        qpd_collector.append(_qpd.data)
        _spi = SpaceRangerImporter(paths=bp.spcrng_paths, use_hires_img=bp.use_qp_hes)
        _df = _spi.load_data(filtered_data=True)

        _hpm = HistoPathMerger(masks=_masks, spots=_spi.spots,
                               exclusive_mask_pairs=_excl_mask_pairs, qp_data=_qpd.data)
        df_collector.append(_hpm.merge())
        spots_collector.append(_hpm.spots)
        if pickle_spots:
            SpaceRangerImporter.pickle_spots(_hpm.spots, os.path.join(output_folder, 'pickled_spots'))
        logger.info("Done with {}".format(bp.id))
    return spots_collector, qpd_collector, df_collector