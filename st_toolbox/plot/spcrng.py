import logging
from typing import Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum, unique
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
import cv2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .utils import auto_split_contours
from ..spcrng.spacerange_import import SpaceRangerSpots, SpaceRangerSpot
from ..qupath.data_importer import QuPathDataObject, AnnotationObject

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)


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
    """
    :param name: the name of the filer
    :param cutoff: float
    :param type: CutoffType enum
    """
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
    #return int(c_rgb[2]), int(c_rgb[1]), int(c_rgb[0])


def generate_annotation_colors(qp_data: QuPathDataObject, cm: str='hsv') -> Dict[str, Tuple[float, float, float, float]]:
    a_colors = {}
    for a in qp_data.annotations:
        if a.name not in a_colors:
            a_colors[a.name] = (0, 0, 0)
    n_colors = len(a_colors)
    cm = plt.get_cmap(cm, n_colors+1)
    # now get the normalized values
    norm = mpl.colors.Normalize(vmin=0, vmax=n_colors)
    for index, (name, c) in enumerate(sorted(a_colors.items())):
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


def get_spots_img(spots: SpaceRangerSpots,
                  data: pd.Series,
                  cm='viridis',
                  external_cm: mpl.colors.Colormap = None,
                  external_norm: np.ndarray = None,
                  log2transform: bool = True,
                  spot_filters: List[Filter] = None,
                  spot_filter_condition: FilterCondition = FilterCondition.ALL_TRUE) -> Tuple[np.ndarray, mpl.colors.Colormap, np.ndarray]:
    if log2transform:
        reads = log2_transform(data)
    else:
        reads = data
    # TODO check if that many colors is a smart idea
    if external_cm is None or external_norm is None:
        n_colors = int(np.ceil(np.max(reads) - np.min(reads)))
        cm = plt.get_cmap(cm, n_colors)
        # now get the normalized values
        norm = mpl.colors.Normalize(vmin=np.min(reads), vmax=np.max(reads))
    img = spots.img.copy()
    # draw spots with cv2 - more accurate!
    logger.info("drawing spots")
    for spot in spots:
        c = convert_color(cm(norm(reads[spot.unique_name])))
        if spot_filters is None or check_spot_filters(spot, spot_filters, spot_filter_condition):
            img = cv2.circle(img=img,
                             center=(int(round(spot.img_col)), int(round(spot.img_row))),
                             radius=int(round(spot.diameter_px / 2)),
                             color=c,
                             thickness=-1)
    return img, cm, norm


def generate_annotations_legend(annotation_colors: Dict[str, Tuple[float, float, float, float]],
                                omit_annotations: List[str] = None,
                                draw_only_annotations: List[str] = None) -> Tuple[List[mpl.lines.Line2D], List[str]]:
    a_legend_lines = []
    a_legend_labels = []
    for name, color in sorted(annotation_colors.items()):
        if (draw_only_annotations is None or name in draw_only_annotations) and (omit_annotations is None or name not in omit_annotations):
            a_legend_labels.append(name)
            a_legend_lines.append(mpl.lines.Line2D([0], [0], color=color, lw=4))
    return a_legend_lines, a_legend_labels


def draw_annotations_on_img(qp_data: QuPathDataObject,
                            img: np.ndarray,
                            scalefactor: float,
                            annotation_colors: Dict[str, Tuple[float, float, float, float]] = None,
                            auto_split_distance: float = 20.,
                            omit_annotations: List[str] = None,
                            draw_only_annotations: List[str] = None) -> Tuple[Dict[str, Tuple[float, float, float, float]], np.ndarray]:
    # TODO make this better
    if qp_data is not None:
        if annotation_colors is None:
            annotation_colors = generate_annotation_colors(qp_data)
        logger.info("drawing annotations")
        for a, c in annotation_colors.items():
            logger.debug("annotations_color for {}: rgba {} - converted cv2 {}".format(a, c, (convert_color(c))))
        for a in qp_data.annotations:
            logger.debug("annotation {}".format(a.name))
            if (draw_only_annotations is None or a.name in draw_only_annotations) and (omit_annotations is None or a.name not in omit_annotations):
                if a.name in annotation_colors:
                    _c = convert_color(annotation_colors[a.name])
                else:
                    _c = (0, 0, 0)
                img = cv2.drawContours(img, auto_split_contours(
                    np.round(a.shape.contour * scalefactor).astype(np.int32),
                    max_distance=auto_split_distance), -1, _c, 2, cv2.LINE_AA)
        return annotation_colors, img


def draw_img_on_axis(ax: mpl.axes.Axes,
                     img: np.ndarray,
                     data: pd.DataFrame,
                     draw_padding: int = None,
                     auto_size_filter: Filter = None,
                     annotations_legend: Tuple[List[mpl.lines.Line2D], List[str]] = None,
                     external_scalefactor: float = None):
    if draw_padding is None:
        offset = 100
    else:
        offset = int(draw_padding)
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
    _img = img[h_min:h_max, w_min:w_max, :]
    if external_scalefactor is not None:
        # some filtering of pointless resizing
        if round(external_scalefactor, 5) != 1:
            _img = cv2.resize(_img, None, fx=external_scalefactor, fy=external_scalefactor, interpolation=cv2.INTER_AREA)
        else:
            logger.warning("omitted scaling by scalefactor close to 1 (sf: {})".format(external_scalefactor))
    ax.imshow(_img)
    #ax.set_ylim(h_max, h_min)
    #ax.set_xlim(w_min, w_max)
    if annotations_legend is not None:
        ax.legend(annotations_legend[0], annotations_legend[1])


def draw_axis(ax: mpl.axes.Axes,
              spots: SpaceRangerSpots,
              gene_col_name: str,
              data: pd.DataFrame = None,
              qp_data: QuPathDataObject = None,
              draw_annotations: bool = True,
              draw_annotations_legend: bool = True,
              annotation_colors: Dict[str, Tuple[float, float, float, float]] = None,
              omit_annotations: List[str] = None,
              draw_only_annotations: List[str] = None,
              cm: str ='viridis',
              log2transform: bool = True,
              auto_split_distance: float = 20.,
              draw_padding: int = None,
              auto_size_filter: Filter = None,
              spot_filters: List[Filter] = None,
              spot_filter_condition: FilterCondition = FilterCondition.ALL_TRUE,
              external_scalefactor: float = None) -> Tuple[np.ndarray, mpl.colors.Colormap, np.ndarray]:

    img, cm, norm = get_spots_img(spots=spots,
                                  data=data[gene_col_name],
                                  cm=cm,
                                  log2transform=log2transform,
                                  spot_filters=spot_filters,
                                  spot_filter_condition=spot_filter_condition)

    if qp_data is not None and draw_annotations:
        annotation_colors, img = draw_annotations_on_img(qp_data=qp_data,
                                                         img=img,
                                                         scalefactor=spots[0].img_scalefactor,
                                                         annotation_colors=annotation_colors,
                                                         auto_split_distance=auto_split_distance,
                                                         omit_annotations=omit_annotations,
                                                         draw_only_annotations=draw_only_annotations)
    if draw_annotations_legend:
        annotations_legend = generate_annotations_legend(annotation_colors=annotation_colors,
                                                         omit_annotations=omit_annotations,
                                                         draw_only_annotations=draw_only_annotations)
    else:
        annotations_legend = None

    draw_img_on_axis(ax=ax,
                     img=img,
                     data=data,
                     draw_padding=draw_padding,
                     auto_size_filter=auto_size_filter,
                     annotations_legend=annotations_legend,
                     external_scalefactor=external_scalefactor)
    return img, cm, norm


def get_universal_cm_and_norm(spots_list: List[SpaceRangerSpots],
                              gene_col_name: str,
                              cm: str = 'viridis') -> Tuple[mpl.colors.Colormap, np.ndarray]:
    series_list = []
    for spots in spots_list:
        if gene_col_name in spots.df:
            series_list.append(spots.df[gene_col_name])
        else:
            logger.error("{} not in {}".format(gene_col_name, spots.name))
            return None
    reads = pd.concat(series_list)
    n_colors = int(np.ceil(np.max(reads) - np.min(reads)))
    cm = plt.get_cmap(cm, n_colors)
    # now get the normalized values
    norm = mpl.colors.Normalize(vmin=np.min(reads), vmax=np.max(reads))
    return cm, norm


def get_spot_diameter(spots_list: List[SpaceRangerSpots], condition: str = 'min') -> int:
    diameters = []
    for spots in spots_list:
        diameters.append(spots[0].diameter_px)
    if condition == 'min':
        return np.min(diameters)
    if condition == 'max':
        return np.max(diameters)
    if condition == 'mean':
        return int(round(np.mean(diameters)))
    else:
        return np.min(diameters)


def make_figure(spots: SpaceRangerSpots, gene_name: str, fig_size: Tuple[int, int] = (20,20),
                external_data: pd.DataFrame = None,
                qp_data: QuPathDataObject = None,
                draw_annotations: bool = True, annotation_colors: Dict[str, Tuple[float, float, float, float]] = None,
                draw_annotation_legend: bool = True,
                omit_annotations: List[str] = None,
                draw_only_annotations: List[str] = None,
                cm='viridis', log2transform: bool = True,
                auto_split_distance: float = 20.,
                draw_padding: int = None,
                auto_size_filter: Filter = None,
                spot_filters: List[Filter] = None,
                spot_filter_condition: FilterCondition = FilterCondition.ALL_TRUE) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:

    if external_data is None:
        data = spots.df
    else:
        data = external_data
    if qp_data is None and spots.qp_data.name != '':
        qp_data = spots.qp_data
    # check if gene_name ok
    gene_col_name = get_full_col_name(gene_name=gene_name, data=data)
    if gene_col_name:
        fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
        _, cm, norm = draw_axis(ax=ax1,
                                spots=spots,
                                data=data,
                                gene_col_name=gene_col_name,
                                qp_data=qp_data,
                                draw_annotations=draw_annotations,
                                draw_annotations_legend=draw_annotation_legend,
                                annotation_colors=annotation_colors,
                                omit_annotations=omit_annotations,
                                draw_only_annotations=draw_only_annotations,
                                cm=cm,
                                log2transform=log2transform,
                                auto_split_distance=auto_split_distance,
                                draw_padding=draw_padding,
                                auto_size_filter=auto_size_filter,
                                spot_filters=spot_filters,
                                spot_filter_condition=spot_filter_condition)
        """img, cm, norm = get_spots_img(spots=spots,
                                      data=data[gene_col_name],
                                      cm=cm,
                                      log2transform=log2transform,
                                      spot_filters=spot_filters,
                                      spot_filter_condition=spot_filter_condition)

        annotations_legend = None
        if qp_data is not None and draw_annotations:
            annotation_colors, img = draw_annotations_on_img(qp_data=qp_data,
                                                                           img=img,
                                                                           scalefactor=spots[0].img_scalefactor,
                                                                           annotation_colors=annotation_colors,
                                                                           auto_split_distance=auto_split_distance,
                                                                           omit_annotations=omit_annotations,
                                                                           draw_only_annotations=draw_only_annotations)
            annotations_legend = (a_legend_lines, a_legend_labels)

        

        draw_img_on_axis(ax=ax1,
                         img=img,
                         data=data,
                         draw_padding=draw_padding,
                         auto_size_filter=auto_size_filter,
                         annotations_legend=annotations_legend)"""
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
        logger.warning('could not find a column matching {} - I\'ll check what find_similar spits out for you: {}'.format(
            gene_name, find_similar(gene_name, data)))


def make_one_legend(spots_list: List[SpaceRangerSpots],
                    cm: str = 'hsv',
                    omit_annotations: List[str]=None,
                    draw_only_annotations: List[str] = None,
                    annotation_colors: Dict[str, Tuple[float, float, float, float]] = None):
    # one legend
    _cat_qp_data = QuPathDataObject(
        name='cat_qp_data',
        img_file_base_name='None',
        downsample=1.,
        org_px_height_micron=1.,
        org_px_width_micron=1.,
        org_px_avg_micron=1.,
        cells=[],
        annotations=[]
    )
    for spots in spots_list:
        if spots.qp_data.name != '':
            _cat_qp_data.annotations.extend(spots.qp_data.annotations)
    if len(_cat_qp_data.annotations) > 0:
        # TODO build in a failsafe
        if annotation_colors is None:
            annotation_colors = generate_annotation_colors(qp_data=_cat_qp_data, cm=cm)
        legend = generate_annotations_legend(annotation_colors=annotation_colors,
                                             omit_annotations=omit_annotations,
                                             draw_only_annotations=draw_only_annotations)
    return legend, annotation_colors


def make_figure_grid(spots_list: List[SpaceRangerSpots],
                     gene_name: str,
                     fig_size: Tuple[int, int] = (20,20),
                     draw_annotations: bool = True,
                     annotation_colors: Dict[str, Tuple[float, float, float, float]] = None,
                     annotations_cm: str = 'hsv',
                     omit_annotations: List[str] = None,
                     draw_only_annotations: List[str] = None,
                     external_subfigure_titles: List[str] = None,
                     draw_axes_titles: bool = True,
                     cm: str='viridis',
                     log2transform: bool = True,
                     auto_split_distance: float = 20.,
                     axes_pad: float = 0.05,
                     axes_invisible = True,
                     nrows_ncols: Tuple[int, int] = None,
                     share_all: bool = False,
                     aspect: bool = True,
                     legend_bbox_to_anchor: Tuple[float, float] = (1.3, 1),
                     legend_loc: str = 'upper left',
                     legend_borderaxespad: float = 0.,
                     cbar_location: str = 'right',
                     cbar_mode: str = 'single',
                     cbar_size: str = '5%',
                     cbar_pad: float = 0.05,
                     draw_padding: int = None,
                     auto_size_filter: Filter = None,
                     spot_filters: List[Filter] = None,
                     spot_filter_condition: FilterCondition = FilterCondition.ALL_TRUE) -> Tuple[mpl.figure.Figure, ImageGrid]:

        n_subfigs = len(spots_list)
        if external_subfigure_titles is not None and len(external_subfigure_titles) == n_subfigs:
            ax_titles = external_subfigure_titles
        else:
            ax_titles = [s.name for s in spots_list]
        if nrows_ncols is None:
            nrows_ncols = (int(round(np.sqrt(n_subfigs))), int(np.ceil(np.sqrt(n_subfigs))))

        if draw_annotations:
            legend, annotation_colors = make_one_legend(spots_list=spots_list,
                                                        cm=annotations_cm,
                                                        omit_annotations=omit_annotations,
                                                        draw_only_annotations=draw_only_annotations,
                                                        annotation_colors=annotation_colors)
        else:
            legend = None
        if draw_axes_titles and axes_pad < 0.25:
            logger.warning("axes_pad of {} to small for use with axis titles - should be >= 0.25 - auto set to 0.25 now".format(axes_pad))
            axes_pad = 0.25
        fig = plt.figure(figsize=fig_size)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=nrows_ncols,
                         axes_pad=axes_pad,
                         share_all=share_all,
                         aspect=aspect,
                         cbar_location=cbar_location,
                         cbar_mode=cbar_mode,
                         cbar_size=cbar_size,
                         cbar_pad=cbar_pad)
        min_spot_diameter = get_spot_diameter(spots_list=spots_list, condition='min')
        for grid_ax, spots, ax_label in zip(grid, spots_list, ax_titles):
            data = spots.df
            if spots.qp_data.name != '':
                qp_data = spots.qp_data
            else:
                qp_data = None
            gene_col_name = get_full_col_name(gene_name=gene_name, data=data)
            if gene_col_name:
                sf = min_spot_diameter / spots[0].diameter_px
                _, cm, norm = draw_axis(ax=grid_ax,
                                        spots=spots,
                                        data=data,
                                        gene_col_name=gene_col_name,
                                        qp_data=qp_data,
                                        draw_annotations=draw_annotations,
                                        draw_annotations_legend=False,
                                        annotation_colors=annotation_colors,
                                        omit_annotations=omit_annotations,
                                        draw_only_annotations=draw_only_annotations,
                                        cm=cm,
                                        log2transform=log2transform,
                                        auto_split_distance=auto_split_distance,
                                        draw_padding=draw_padding,
                                        auto_size_filter=auto_size_filter,
                                        spot_filters=spot_filters,
                                        spot_filter_condition=spot_filter_condition,
                                        external_scalefactor=sf)
                if draw_axes_titles:
                    grid_ax.set_title(ax_label)
                if axes_invisible:
                    grid_ax.axis('off')
            else:
                logger.warning(
                    'could not find a column matching {} - I\'ll check what find_similar spits out for you: {}'.format(
                        gene_name, find_similar(gene_name, data)))
                return None
        if cbar_mode == 'single':
            cm, norm = get_universal_cm_and_norm(spots_list=spots_list, gene_col_name=gene_col_name, cm=cm)
            if log2transform:
                cbticks = mpl.ticker.LogLocator(base=2.0)
                cblabel = '{} log2-fold change'.format(gene_name.replace('_', ' '))
            else:
                cbticks = None
                cblabel = '{} reads'.format(gene_name.replace('_', ' '))
            cb = mpl.colorbar.ColorbarBase(grid.cbar_axes[0],
                                           cmap=cm,
                                           norm=norm,
                                           ticks=cbticks,
                                           label=cblabel,
                                           orientation='vertical')

        # legend
        if legend is not None:
            grid.cbar_axes[0].legend(handles=legend[0], labels=legend[1],
                                     bbox_to_anchor=legend_bbox_to_anchor,
                                     loc=legend_loc,
                                     borderaxespad=legend_borderaxespad)
        return fig, grid


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