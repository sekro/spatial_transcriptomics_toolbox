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
from st_toolbox.qupath import QuPathBinMaskImporter, QuPathDataImporter, QuPathDataObject
from st_toolbox.qupath.coregistration import CoRegistrationData
from st_toolbox.spcrng import SpaceRangerPaths, SpaceRangerImporter, SpaceRangerSpots
from st_toolbox.spcrng.merge_data import HistoPathMerger, ExclusiveMaskPair

GLOBAL_debug = True
logger = logging.getLogger(__name__)

if GLOBAL_debug:
    logging.basicConfig(level=logging.DEBUG)

@dataclass
class BatchPaths:
    id: str
    qp_masks: List[str]
    qp_data: str
    spcrng_paths: SpaceRangerPaths
    coreg_data: CoRegistrationData


def batch_merge(batches: List[BatchPaths], output_folder: str,
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

        _qpms = QuPathBinMaskImporter.batch_import(qp_export_path_list=bp.qp_masks,
                                                   output_folder=os.path.join(output_folder, 'qp_mask_import'),
                                                   co_registration_data_list=[bp.coreg_data for i in range(0, len(bp.qp_masks))])
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
                                  co_registration_data=bp.coreg_data)
        _qpd.run()
        qpd_collector.append(_qpd.data)
        _spi = SpaceRangerImporter(paths=bp.spcrng_paths)
        _df = _spi.load_data(filtered_data=True)
        spots_collector.append(_spi.spots)
        _hpm = HistoPathMerger(masks=_masks, spots=_spi.spots,
                               exclusive_mask_pairs=_excl_mask_pairs, qp_data=_qpd.data)
        df_collector.append(_hpm.merge())
        logger.info("Done with {}".format(bp.id))
    return spots_collector, qpd_collector, df_collector