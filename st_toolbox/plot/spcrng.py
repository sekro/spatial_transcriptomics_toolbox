import logging
from typing import Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum, unique
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from ..spcrng.spacerange_import import SpaceRangerSpots, SpaceRangerSpot
from ..qupath.data_importer import QuPathDataObject, AnnotationObject

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@unique
class CutoffType(Enum):
    EQUAL = 0
    GREATER = 1
    LESS = 2
    GREATER_OR_EQUAL = 3
    LESS_OR_EQUAL = 4


@unique
class FilterCondition(Enum):
    ALL_TRUE = 0
    ANY_TRUE = 1
    NONE_TRUE = 2


@dataclass
class Filter:
    name: str
    cutoff: float
    type: CutoffType


# look up gene name using parts of the name
def find_similar(gene_name, data):
    result = data.columns[np.char.find(data.columns.to_list(), gene_name) != -1]
    if len(result) > 0:
        return result
    else:
        print('could not find original str {} - desperatly trying with {}'.format(gene_name, gene_name.upper()))
        return data.columns[np.char.find(data.columns.to_list(), gene_name.upper()) != -1]


# small helper function to get the full col name from the Gene name (provide in caps)
def get_full_col_name(gene_name, data):
    result = data.columns[np.char.find(data.columns.to_list(), gene_name) != -1]
    if len(result) > 0:
        if len(result) > 1:
            print('Your gene name returned several cols: {} - returning only the first hit {}'.format(result,
                                                                                                      result[0]))
        return result[0]
    else:
        return ''


def get_image(dataset_name, images_dict):
    if dataset_name in images_dict:
        return cv2.cvtColor(cv2.imread(images_dict[dataset_name], cv2.IMREAD_COLOR), cv2.COLOR_BGRA2RGB)
    else:
        print('WARNING image for {} not found'.format(dataset_name))
        return None


def log2_transform(x: pd.Series, zero_gets: float=-1.) -> pd.Series:
    x[x == 0] = np.nan
    return np.log2(x).fillna(zero_gets)


def convert_color(color):
    c_rgb = np.round(np.array([color[0], color[1], color[2]]) * 255)
    return int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2])


def generate_annotation_colors(qp_data: QuPathDataObject, cm: str='hsv') -> Dict[str, Tuple[float, float, float, float]]:
    a_colors = {}
    for a in qp_data.annotations:
        a_colors[a.name] = (0, 0, 0)
    n_colors = len(a_colors)
    cm = plt.get_cmap(cm, n_colors+1)
    # now get the normalized values
    norm = mpl.colors.Normalize(vmin=0, vmax=n_colors)
    for index, name in enumerate(a_colors.keys()):
        a_colors[name] = cm(norm(index))
    return a_colors


def filter_type_check(filter: Filter, value) -> bool:
    if filter.type == CutoffType.EQUAL:
        return filter.cutoff == value
    elif filter.type == CutoffType.GREATER:
        return value > filter.cutoff
    elif filter.type == CutoffType.LESS:
        return value < filter.cutoff
    elif filter.type == CutoffType.GREATER_OR_EQUAL:
        return value >= filter.cutoff
    elif filter.type == CutoffType.LESS_OR_EQUAL:
        return value <= filter.cutoff
    else:
        return False


def check_spot_filters(spot: SpaceRangerSpot, filters: List[Filter], condition: FilterCondition) -> bool:
    checks = []
    for f in filters:
        if f.name in spot.__dict__:
            checks.append(filter_type_check(f, spot.__dict__[f.name]))
        elif isinstance(spot.metadata, pd.Series) and f.name in spot.metadata.index:
            checks.append(filter_type_check(f, spot.metadata[f.name]))
        elif isinstance(spot.reads, pd.Series) and f.name in spot.reads.index:
            checks.append(filter_type_check(f, spot.reads[f.name]))
        else:
            checks.append(False)
    if condition == FilterCondition.ALL_TRUE:
        return all(checks)
    elif condition == FilterCondition.ANY_TRUE:
        return any(checks)
    elif condition == FilterCondition.NONE_TRUE:
        return not any(checks)
    else:
        return False


def make_figure(spots: SpaceRangerSpots, gene_name: str, fig_size: Tuple[int, int]=(20,20),
                external_data: pd.DataFrame=None,
                qp_data: QuPathDataObject=None,
                draw_annotations: bool=True, annotation_colors: Dict[str, Tuple[float, float, float, float]]=None,
                omit_annotations: List[str]=None,
                cm='viridis', log2transform: bool=True,
                auto_size_filter: Filter=None,
                spot_filters: List[Filter]=None,
                spot_filter_condition: FilterCondition=FilterCondition.ALL_TRUE) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    if external_data is None:
        data = spots.df
    else:
        data = external_data
    if qp_data is None and spots.qp_data.name != '':
        qp_data = spots.qp_data
    # check if gene_name ok
    gene_col_name = get_full_col_name(gene_name=gene_name, data=data)
    if gene_col_name:
        if log2transform:
            reads = log2_transform(data[gene_col_name])
        else:
            reads = data[gene_col_name]
        # TODO check if that many colors is a smart idea
        max_reads = data[gene_col_name].max()
        min_reads = data[gene_col_name].min()
        n_colors = max_reads - min_reads
        cm = plt.get_cmap(cm, n_colors)
        # now get the normalized values
        norm = mpl.colors.Normalize(vmin=np.min(reads), vmax=np.max(reads))
        img = spots.img.copy()
        height, width, c = img.shape

        # draw spots with cv2 - more accurate!
        logger.info("drawing spots")
        for spot in spots:
            c = convert_color(cm(norm(reads[spot.unique_name])))
            if spot_filters is None or check_spot_filters(spot, spot_filters, spot_filter_condition):
                img = cv2.circle(img=img,
                                 center=(int(round(spot.img_col)), int(round(spot.img_row))),
                                 radius=int(round(spot.diameter_px/2)),
                                 color=c,
                                 thickness=-1)

        # TODO make this better
        annotations_legend = None
        if qp_data is not None and draw_annotations:
            a_legend_lines = []
            a_legend_labels = []
            if annotation_colors is None:
                annotation_colors = generate_annotation_colors(qp_data)
            logger.info("drawing annotations")
            for a in qp_data.annotations:
                logger.debug("annotation {}".format(a.name))
                if omit_annotations is None or a.name not in omit_annotations:
                    logger.debug("draw this annotation")
                    a_legend_labels.append(a.name)
                    a_legend_lines.append(mpl.lines.Line2D([0], [0], color=annotation_colors[a.name], lw=4))
                    # TODO need to consider implementing own draw function / or split "multi-contour-contours" to avoid connecting lines between contours
                    img = cv2.drawContours(img, [np.round(a.shape.contour * spots[0].img_scalefactor).astype(np.int32)], 0, convert_color(annotation_colors[a.name]), 2)
            annotations_legend = (a_legend_lines, a_legend_labels)


        fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
        offset = 50
        if auto_size_filter is not None and auto_size_filter.name in data.columns:
            col_min = data[filter_type_check(auto_size_filter, data[auto_size_filter.name])]['img_col'].min()
            col_max = data[filter_type_check(auto_size_filter, data[auto_size_filter.name])]['img_col'].max()
            row_min = data[filter_type_check(auto_size_filter, data[auto_size_filter.name])]['img_row'].min()
            row_max = data[filter_type_check(auto_size_filter, data[auto_size_filter.name])]['img_row'].max()
        else:
            col_min = data['img_col'].min()
            col_max = data['img_col'].max()
            row_min = data['img_row'].min()
            row_max = data['img_row'].max()
        w_min = int(col_min) - offset
        w_max = int(col_max) + offset
        h_min = int(row_min) - offset
        h_max = int(row_max) + offset
        ax1.imshow(img)
        ax1.set_ylim(h_max, h_min)
        ax1.set_xlim(w_min, w_max)
        if annotations_legend is not None:
            ax1.legend(annotations_legend[0], annotations_legend[1])
        axins = inset_axes(ax1,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax1.transAxes,
                           borderpad=0,
                           )
        cb1 = mpl.colorbar.ColorbarBase(axins, cmap=cm,
                                        norm=norm,
                                        orientation='vertical')
        if log2transform:
            cb1.set_label('{} log2-fold change'.format(gene_name.replace('_', ' ')))
        else:
            cb1.set_label('{} reads'.format(gene_name.replace('_', ' ')))
        return fig, ax1
    else:
        print('could not find a column matching {} - I\'ll check what find_similar spits out for you: {}'.format(
            gene_name, find_similar(gene_name, data)))


def img_of(dataset_name, gene_name, datasets, scalefactors, images_dict, mask=None, mask_cut_off=0.5,
           figsize_u=(20, 20), vis_tune_factor=1):
    if (np.char.find(datasets.index.to_list(), dataset_name) != -1).any():
        data = datasets.loc[datasets.index[np.char.find(datasets.index.to_list(), dataset_name) != -1]]
        scalefactors = scalefactors[dataset_name]
        if gene_name == 'umi':
            colname = gene_name
        else:
            colname = get_full_col_name(gene_name, data)
        if colname:
            image = get_image(dataset_name, images_dict)
            height, width, c = image.shape

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_u, sharex='all', sharey='all')
            offset = 50
            w_min = int(data['imagecol'].min()) - offset
            w_max = int(data['imagecol'].max()) + offset
            h_min = int(data['imagerow'].min()) - offset
            h_max = int(data['imagerow'].max()) + offset
            ax1.imshow(image)
            ax2.imshow(image)
            ax1.set_ylim(h_max, h_min)
            ax2.set_xlim(w_min, w_max)
            # scalefactors['spot_diameter_fullres']
            # spots not to scale!
            actual_fig_px_u = (h_max - h_min) / figsize_u[0]
            max_fig_px_u = height / figsize_u[0]
            spot_factor = (h_max - h_min) / height
            # spotsize - data units to points - see
            # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale
            # https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
            spot_size = vis_tune_factor * (
                        ax2.get_window_extent().height / (h_max - h_min) * 72 / fig.dpi * scalefactors[
                    'spot_diameter_fullres'] / 2 * scalefactors['tissue_hires_scalef']) ** 2

            if mask is not None:
                data[data[mask] > mask_cut_off].plot.scatter(x='imagecol', y='imagerow', s=spot_size, c=colname,
                                                             colormap='viridis', ax=ax2, colorbar=False)
            else:
                data.plot.scatter(x='imagecol', y='imagerow', s=spot_size, c=colname, colormap='viridis', ax=ax2,
                                  colorbar=False)
            axins = inset_axes(ax2,
                               width="5%",  # width = 5% of parent_bbox width
                               height="100%",  # height : 50%
                               loc='lower left',
                               bbox_to_anchor=(1.05, 0., 1, 1),
                               bbox_transform=ax2.transAxes,
                               borderpad=0,
                               )
            cmap = mpl.cm.viridis
            norm = mpl.colors.Normalize(vmin=data[colname].min(), vmax=data[colname].max())
            cb1 = mpl.colorbar.ColorbarBase(axins, cmap=cmap,
                                            norm=norm,
                                            orientation='vertical')
            cb1.set_label('{} reads'.format(gene_name))
            return fig, ax1, ax2
        else:
            print('could not find a column matching {} - I\'ll check what find_similar spits out for you: {}'.format(
                gene_name, find_similar(gene_name, data)))
    else:
        print('dataset not in data')