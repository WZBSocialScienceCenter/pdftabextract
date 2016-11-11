# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:40:39 2016

@author: mkonrad
"""

from collections import defaultdict, OrderedDict
from logging import warning

import json
import csv

from scipy.cluster.hierarchy import fclusterdata
import numpy as np

from pdftabextract.geom import pt, rect, rectintersect, rectcenter_dist
from pdftabextract.common import read_xml, parse_pages, get_bodytexts, divide_texts_horizontally, sorted_by_attr, \
                                 texts_at_page_corners, mode, flatten_list


_conf = {}  # global configuration settings


# TODO: real logging instead of print()

#%%

class JSONEncoderPlus(json.JSONEncoder):
    """
    JSON encode also for iterables
    Needed for save_page_grids()
    """
    def default(self, o):
       try:
           iterable = iter(o)
       except TypeError:
           pass
       else:
           return list(iterable)
       # Let the base class default method raise the TypeError
       return super().default(self, o)


#%%
def set_config(c):
    global _conf
    _conf = c


def set_config_option(o, v):
    _conf[o] = v


def extract_tabular_data_from_xmlfile(xmlfile):
    subpages = get_subpages_from_xmlfile(xmlfile)
    
    return extract_tabular_data_from_subpages(subpages)


def extract_tabular_data_from_xmlroot(xmlroot, grids_as_list=False):
    subpages = get_subpages_from_xmlroot(xmlroot)
    
    return extract_tabular_data_from_subpages(subpages, grids_as_list=grids_as_list)


def extract_tabular_data_from_subpages(subpages, grids_as_list=False):
    skip_pages = _conf.get('skip_pages', [])
    
    # analyze the row/column layout of each page and return these layouts,
    # a list of invalid subpages (no tabular layout could be recognized),
    # and a list of common column positions
    layouts, col_positions, mean_row_height, page_column_offsets = analyze_subpage_layouts(subpages)
        
    # go through all subpages
    output = OrderedDict()
    page_grids = OrderedDict()
    page_gridlists = OrderedDict()
    skipped_pages = []
    for p_id in sorted(subpages.keys(), key=lambda x: x[0]):
        if p_id in skip_pages:
            continue
        
        sub_p = subpages[p_id] 
        page_col_offset = page_column_offsets[p_id]
        
        if abs(page_col_offset) > _conf.get('max_page_col_offset_thresh', 0):
            page_col_offset = 0
        
        pagenum, pageside = p_id
        
        if pagenum not in output:
            output[pagenum] = OrderedDict()
        if pagenum not in page_grids:
            page_grids[pagenum] = OrderedDict()
        if pagenum not in page_gridlists:
            page_gridlists[pagenum] = OrderedDict()
        
        # get the column positions
        subp_col_positions = list(np.array(col_positions[pageside]) + sub_p['x_offset'] + page_col_offset)
        
        # get the row positions        
        row_positions = layouts[p_id][0]
        if not row_positions:
            row_positions, guessed_row_height = guess_row_positions(sub_p, mean_row_height, subp_col_positions)
            if row_positions:
                print("subpage %d/%s layout: guessed %d rows (mean row height %f, guessed row height %f)"
                      % (sub_p['number'], sub_p['subpage'], len(row_positions), mean_row_height, guessed_row_height))
        
        if not row_positions:
            print("subpage %d/%s layout: no row positions identified -- skipped" % (sub_p['number'], sub_p['subpage']))
            skipped_pages.append(p_id)
            continue
        
        # fit the textboxes from this subpage into the tabular grid defined by the
        # row and column positions
        table, grid, grid_list = create_datatable_from_subpage(sub_p,
                                                               row_positions=row_positions,
                                                               col_positions=subp_col_positions,
                                                               grid_as_list=grids_as_list)

        # create a table of texts from the textboxes inside the table cells
        table_texts = []
        for row in table:
            row_texts = []
            for cell in row:
                # form lines from the textboxes in this cell
                cell_lines = put_texts_in_lines(cell)
                row_texts.append(create_text_from_lines(cell_lines))
            table_texts.append(row_texts)
        
        # add this table to the output
        output[pagenum][pageside] = table_texts
        page_grids[pagenum][pageside] = grid
        page_gridlists[pagenum][pageside] = grid_list

    if grids_as_list:
        return output, skip_pages + skipped_pages, page_grids, page_gridlists
    else:
        return output, skip_pages + skipped_pages, page_grids


def save_tabular_data_dict_as_json(data, jsonfile):
    with open(jsonfile, 'w') as f:
        json.dump(data, f, indent=2)


def save_tabular_data_dict_as_csv(data, csvfile):
    with open(csvfile, 'w') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        csv_rows = convert_tabular_data_dict_to_matrix(data)
        
        n_addcols = len(csv_rows[0]) - 2
        csv_header = ['page_num', 'page_side'] + ['col' + str(i+1) for i in range(n_addcols)]
        
        csvwriter.writerows([csv_header] + csv_rows)


def save_page_grids(page_grids, jsonfile):
    with open(jsonfile, 'w') as f:
        json.dump(page_grids, f, cls=JSONEncoderPlus)


def convert_tabular_data_dict_to_matrix(data):
    rows = []
    for p_num, subpages in data.items():
        for p_side, table_texts in subpages.items():
            for table_row in table_texts:
                rows.append([p_num, p_side] + table_row)
    
    return rows


def get_subpages_from_xmlfile(xmlfile):
    _, root = read_xml(xmlfile)
    
    return get_subpages_from_xmlroot(root)


def get_subpages_from_xmlroot(xmlroot):
    pages = parse_pages(xmlroot)

    subpages = {}
    
    for p_num, page in pages.items():
        # strip off footer and header
        bodytexts = get_bodytexts(page, _conf.get('header_skip', 0), _conf.get('footer_skip', 0))
        
        if _conf.get('divide', 0) != 0:
            page_subpages = divide_texts_horizontally(page, _conf.get('divide'), bodytexts)
        else:
            singlepage = dict(page)
            singlepage['texts'] = bodytexts
            page_subpages = (singlepage, )
            
        for sub_p in page_subpages:
            p_id = (sub_p['number'], sub_p['subpage'])        
            subpages[p_id] = sub_p
    
    return subpages


def guess_row_positions(subpage, mean_row_height, col_positions):
    if not len(col_positions) > 1:
        raise ValueError('number of detected columns must be at least 2')
    
    guess_rows_cond_fns = _conf.get('guess_rows_cond_fns', None)
    
    texts_by_y = sorted_by_attr(subpage['texts'], 'top')
    
    # borders of the first column
    first_col_left, first_col_right = col_positions[0:2]
    
    # additional condition functions
    if guess_rows_cond_fns:
        cond_fn_top, cond_fn_bottom = guess_rows_cond_fns
    else:
        cond_fn_top, cond_fn_bottom = None, None
    
    # find the first (top-most) text that fits in a possible table cell in the first column
    top_text = None        
    for t in texts_by_y:
        t_rect = rect_from_text(t)
        cell_rect = rect(pt(first_col_left, t['top']), pt(first_col_right, t['top'] + mean_row_height))        
        isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
        if isect is not None and isect >= 0.5 and (cond_fn_top is None or cond_fn_top(t)):
            top_text = t
            break
    
    # borders of the last column
    #last_col_left, last_col_right = col_positions[-2:]
    
    # find the last (lowest) text that fits in a possible table cell in the first column
    bottom_text = None
    for t in reversed(texts_by_y):
        t_rect = rect_from_text(t)
        cell_rect = rect(pt(first_col_left, t['top']), pt(first_col_right, t['top'] + mean_row_height))
        isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
        if isect is not None and isect >= 0.5 and (cond_fn_bottom is None or cond_fn_bottom(t)):
            bottom_text = t
            break
    
    if not top_text:
        warning("subpage %d/%s: could not find top text" % (subpage['number'], subpage['subpage']))
        return None, None
    
    if not bottom_text:
        warning("subpage %d/%s: could not find bottom text" % (subpage['number'], subpage['subpage']))
        return None, None

    top_border = int(np.round(top_text['top']))

    fixed_row_num = _conf.get('fixed_row_num', 0)    
    if fixed_row_num:
        bottom_border = min(top_border + mean_row_height * fixed_row_num, subpage['height'])
    else:        
        bottom_border = int(np.round(bottom_text['top'] + mean_row_height))
        
    
    table_height = bottom_border - top_border

    n_rows = round(table_height / mean_row_height)
    
#    n_rows, remainder = divmod(table_height, mean_row_height)
#    
#    if remainder / mean_row_height > 0.5:   # seems like the number of rows doesn't really fit
#        warning("subpage %d/%s: the number of rows doesn't really fit the guessed table height"
#                % (subpage['number'], subpage['subpage']))
#        return None                         # we assume this is an invalid table layout
    
    optimal_row_height = table_height // n_rows
    return list(range(top_border, bottom_border, optimal_row_height))[:n_rows], optimal_row_height


def analyze_subpage_layouts(subpages):
    skip_pages = _conf.get('skip_pages', [])
    pages_divided = _conf.get('divide', 0) > 0
    corner_box_cond_fns = _conf.get('corner_box_cond_fns', None)
    
    # find the column and row borders for each subpage
    layouts = {}
    pages_dims = {}
    for p_id, sub_p in subpages.items():      
        layout = [None, None]
        
        if p_id not in skip_pages:
            try:
                col_positions, row_positions, page_dims = find_col_and_row_positions_in_subpage(sub_p,
                                                                                                corner_box_cond_fns)
            except ValueError as e:
                print("subpage %d/%s layout: skipped ('%s')" % (sub_p['number'], sub_p['subpage'], str(e)))
                col_positions, row_positions = None, None
                page_dims = None
            
            if row_positions:
                layout[0] = row_positions
            
            if col_positions:
                layout[1] = col_positions
                        
            if layout == (None, None):
                print("subpage %d/%s layout: invalid column/row positions: %s/%s"
                      % (sub_p['number'], sub_p['subpage'], col_positions, row_positions))
                layout = None
        
        layouts[p_id] = layout
        pages_dims[p_id] = page_dims
    
    # get the row and column positions of all valid subpages
    all_row_pos = [np.array(layout[0]) for layout in layouts.values() if layout[0]]
    all_col_pos = [(p_id, np.array(layout[1]) - subpages[p_id]['x_offset']) for p_id, layout in layouts.items()
                   if layout[1]]
        
    # get all numbers of rows and columns across the subpages   
    nrows = [len(row_pos) for row_pos in all_row_pos
             if len(row_pos) > _conf.get('best_rows_selection_min_rows_thresh', 0)]
    ncols = [len(col_pos) for _, col_pos in all_col_pos
             if len(col_pos) > _conf.get('best_cols_selection_min_cols_thresh', 0)]
    
    print("row numbers:", nrows)
    print("col numbers:", ncols)
    
    nrows_best = _conf.get('best_rows_selection_fn', mode)(nrows) if nrows else None
    ncols_best = _conf.get('best_cols_selection_fn', mode)(ncols)
    
    print("best number of rows:", nrows_best)
    print("best number of columns:", ncols_best)
    
    # find the best row height
    overall_row_height_mean = _conf.get('fixed_row_height', None)
    if not overall_row_height_mean:
        assert nrows_best is not None
        
        row_pos_w_best_len = [row_pos for row_pos in all_row_pos if len(row_pos) == nrows_best]
        selected_row_pos = [row_pos[2:nrows_best-1] for row_pos in row_pos_w_best_len]
        assert len(selected_row_pos) > 1
        row_height_means = []
        for row_pos in selected_row_pos:
            row_height_means.append(np.mean([pos - row_pos[i-1] for i, pos in enumerate(row_pos[1:])]))
        overall_row_height_mean = int(np.round(np.mean(row_height_means)))
    
    # find the "best" (median) column positions per subpage side
    subpage_sides = ('left', 'right') if pages_divided else (None, )
    best_col_pos_medians = {}
    for side in subpage_sides:
        col_pos_w_best_len = [col_pos for p_id, col_pos in all_col_pos
                              if len(col_pos) == ncols_best and p_id[1] == side]

        best_col_pos = [list() for _ in range(ncols_best)]
        for col_positions in col_pos_w_best_len:
            for i, pos in enumerate(col_positions):
                best_col_pos[i].append(pos)
    
        best_col_pos_medians[side] = [np.median(best_col_pos[i]) for i in range(ncols_best)]
    
    merge_cols_opt = _conf.get('merge_columns', None)
    if merge_cols_opt:
        for side in subpage_sides:
            for from_col, to_col in merge_cols_opt:
                for i in range(from_col, to_col):
                    best_col_pos_medians[side].pop(i + 1)

    split_cols_opt = _conf.get('split_columns', None)

    if split_cols_opt:
        for side in subpage_sides:
            for split_col, split_ratio in split_cols_opt:
                x1 = best_col_pos_medians[side][split_col]
                if split_col + 1 < len(best_col_pos_medians[side]):
                    x2 = best_col_pos_medians[side][split_col + 1]
                else:
                    first_pagenum = sorted(subpages.keys(), key=lambda x: x[0])[0][0]
                    x2 = subpages[(first_pagenum, side)]['width']
                    
                best_col_pos_medians[side].append(x1 + (x2 - x1) * split_ratio)
                best_col_pos_medians[side] = list(sorted(best_col_pos_medians[side]))
                    
    # set offset per page 
    page_column_offsets = {}
    for p_id, page_dims in pages_dims.items():
        if page_dims and abs(page_dims[0][0]) != float('infinity'):
            page_column_offsets[p_id] = page_dims[0][0] - subpages[p_id]['x_offset'] - best_col_pos_medians[p_id[1]][0]
        else:
            page_column_offsets[p_id] = 0

    # get list of subpages without proper layout    
    #invalid_layouts = [p_id for p_id, layout in layouts.items() if layout is None]
    
    print("number of skipped pages:", len(skip_pages))
    #print("number of invalid layouts (including skipped pages):", len(invalid_layouts))
    
    return layouts, best_col_pos_medians, overall_row_height_mean, page_column_offsets

    
def grid_dict_to_list(grid, grid_dims):    
    grid_list = []
    for r in range(grid_dims[0]):
        row = []
        for c in range(grid_dims[1]):
            row.append(grid[(r, c)])
        grid_list.append(row)
    
    return grid_list
    

def create_datatable_from_subpage(subpage, row_positions, col_positions, grid_as_list=False):
    grid, grid_dims = make_grid_from_positions(subpage, row_positions, col_positions)
    
    if grid_as_list:
        grid_list = grid_dict_to_list(grid, grid_dims)
    else:
        grid_list = None
    
    table = fit_texts_into_grid(subpage['texts'], grid)
    
    return table, grid, grid_list


def fit_texts_into_grid(texts, grid, p_id_for_logging=None):
    keys = list(grid.keys())
    n_rows = max(k[0] for k in keys) + 1
    n_cols = max(k[1] for k in keys) + 1
    
    # create an empty table with the found dimensions
    # each cell will have a list with textblocks inside
    table = np.empty((n_rows, n_cols), dtype='object')
    # np.full does not work with with list as fill value so we have to do it like this:
    for j in range(n_rows):
        for k in range(n_cols):
            table[j, k] = []
    
    # iterate through the textblocks of this page
    for t in texts:
        t_rect = rect_from_text(t)   # rectangle of the textbox
        
        # find out the cells with which this textbox rectangle intersects
        cell_isects = []
        for idx, cell_rect in grid.items():
            isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
            if isect is not None and isect > 0:
                cell_isects.append((idx, isect, rectcenter_dist(t_rect, cell_rect)))
        
        if len(cell_isects) > 0:
            # find out the cell with most overlap
            max_isect_val = max([x[1] for x in cell_isects])
            if max_isect_val < 0.5 and p_id_for_logging:
                warning("subpage %s: low best cell intersection value: %f" % (p_id_for_logging, max_isect_val))
            best_isects = list(sorted([x for x in cell_isects if x[1] == max_isect_val], key=lambda x: x[2]))
            best_idx = best_isects[0][0]
            
            # add this textblock to the table at the found cell index
            table[best_idx].append(t)
        else:
            if p_id_for_logging:
                warning("subpage %s: no cell found for textblock '%s'" % (p_id_for_logging, t['value']))
    
    return table


def find_col_and_row_positions_in_subpage(subpage, corner_box_cond_fns=None):
    if corner_box_cond_fns is None:
        corner_box_cond_fns = (None, ) * 4    
    
    if len(subpage['texts']) < 3:
        raise ValueError('insufficient number of texts on subpage %d/%s' % (subpage['number'], subpage['subpage']))
    
    corner_texts = texts_at_page_corners(subpage, corner_box_cond_fns)
    corner_texts = [None if corner_box_cond_fns[i] is None else t for i, t in enumerate(corner_texts)]
    
    left_x1 = corner_texts[0]['left'] if corner_texts[0] else float('infinity')   # top left
    left_x2 = corner_texts[3]['left'] if corner_texts[3] else float('infinity')   # bottom left
    min_x = min(left_x1, left_x2)
    min_x = subpage['x_offset'] if min_x == float('infinity') else min_x
    
    right_x1 = corner_texts[1]['right'] if corner_texts[1] else float('-infinity')   # top right
    right_x2 = corner_texts[2]['right'] if corner_texts[2] else float('-infinity')   # bottom right
    max_x = max(right_x1, right_x2)
    max_x = subpage['x_offset'] + subpage['width'] if max_x == float('-infinity') else max_x
    
    top_y1 = corner_texts[0]['top'] if corner_texts[0] else float('infinity')   # top left
    top_y2 = corner_texts[1]['top'] if corner_texts[1] else float('infinity')   # top right
    min_y = min(top_y1, top_y2)
    min_y = 0 if min_y == float('infinity') else min_y
    
    bottom_y1 = corner_texts[2]['bottom'] if corner_texts[2] else float('-infinity')   # bottom right
    bottom_y2 = corner_texts[3]['bottom'] if corner_texts[3] else float('-infinity')   # bottom left
    max_y = max(bottom_y1, bottom_y2)
    max_y = subpage['height'] if max_y == float('-infinity') else max_y
    
    filtered_texts = [t for t in subpage['texts']
                      if t['left'] >= min_x and t['right'] <= max_x
                      and t['top'] >= min_y and t['bottom'] <= max_y]
                      
    if len(filtered_texts) < 3:
        raise ValueError('subpage %d/%s: insufficient number of texts after filtering by %f <= x <= %f,  %f <= y <= %f]'
                         % (subpage['number'], subpage['subpage'], min_x, max_x, min_y, max_y))
    
    texts_by_x = list(sorted_by_attr(filtered_texts, 'left'))
    texts_by_y = list(sorted_by_attr(filtered_texts, 'top'))
    
    xs = np.array([t['left'] for t in texts_by_x])
    ys = np.array([t['top'] for t in texts_by_y])
    
    default_property_weights = (1, 1)    
    
    x_clust_ind = find_best_pos_clusters(xs, _conf['possible_ncol_range'], 'x',
                                         property_weights=_conf.get('find_col_clust_property_weights',
                                                                    default_property_weights))
    if xs is None or x_clust_ind is None:
        col_positions = None
    else:
        x_clust_w_vals, _, _ = create_cluster_dicts(xs, x_clust_ind)
        x_clust_w_vals = {c: vals for c, vals in x_clust_w_vals.items() if len(vals) > 3}
        col_positions = positions_list_from_clustervalues(x_clust_w_vals.values())
    
    min_cluster_text_height_thresh = _conf.get('find_row_clust_min_cluster_text_height_thresh', float('-infinity'))
    max_cluster_text_height_thresh = _conf.get('find_row_clust_max_cluster_text_height_thresh', float('infinity'))
    mean_dists_range_thresh = _conf.get('find_row_clust_mean_dists_range_thresh', float('infinity'))
    
    if _conf.get('possible_nrow_range', None) is not None:    
        y_clust_ind = find_best_pos_clusters(ys, _conf['possible_nrow_range'], 'y',
                                             sorted_texts=texts_by_y,
                                             property_weights=_conf.get('find_row_clust_property_weights',
                                                                        default_property_weights),
                                             min_cluster_text_height_thresh=min_cluster_text_height_thresh,
                                             max_cluster_text_height_thresh=max_cluster_text_height_thresh,
                                             mean_dists_range_thresh=mean_dists_range_thresh)
    else:
        y_clust_ind = None
        
    if y_clust_ind is None:
        row_positions = None
    else:
        y_clust_w_vals, _, _ = create_cluster_dicts(ys, y_clust_ind)
        row_positions = positions_list_from_clustervalues(y_clust_w_vals.values())
    
    return col_positions, row_positions, ((min_x, max_x), (min_y, max_y))


def make_grid_from_positions(subpage, rowpos, colpos, as_list=False):
    row_ranges = position_ranges(rowpos, subpage['height'])
    col_ranges = position_ranges(colpos, subpage['x_offset'] + subpage['width'])    
    
    # create a grid with rectangles of table cells
    if not as_list:
        grid = {(r_i, c_i): rect(pt(l, t), pt(r, b)) for r_i, (t, b) in enumerate(row_ranges)
                                                     for c_i, (l, r) in enumerate(col_ranges)}
    else:
        grid = [[rect(pt(l, t), pt(r, b)) for l, r in col_ranges]
                                          for t, b in row_ranges]

    return grid, (len(row_ranges), len(col_ranges))


def position_ranges(positions, last_val):
    p = positions + [last_val]
    return [(p[i-1], v) for i, v in enumerate(p) if i > 0]


def positions_list_from_clustervalues(clust_vals, val_filter=min):
    positions = [val_filter(vals) for vals in clust_vals]
    
    return list(sorted(positions))


def create_cluster_dicts(vals, clust_ind, sorted_texts=None):
    assert len(vals) == len(clust_ind)
    if sorted_texts is not None:
        assert len(clust_ind) == len(sorted_texts)
    
    # build dicts with ...        
    clusters_w_vals = defaultdict(list)     # ... cluster -> [values] mapping
    clusters_w_inds = defaultdict(list)     # ... cluster -> [indices] mapping
    clusters_w_texts = defaultdict(list) if sorted_texts is not None else None   # ... cluster -> [texts] mapping    
    
    for i, (v, c) in enumerate(zip(vals, clust_ind)):
        clusters_w_vals[c].append(v)
        clusters_w_inds[c].append(i)
        if sorted_texts is not None:
            clusters_w_texts[c].append(sorted_texts[i])
    
    return clusters_w_vals, clusters_w_inds, clusters_w_texts


def calc_cluster_means(clusters_w_vals):
    # calculate mean position value per cluster
    return {c: np.mean(vals) for c, vals in clusters_w_vals.items()}


def calc_cluster_medians(clusters_w_vals):
    # calculate median position value per cluster
    return {c: np.median(vals) for c, vals in clusters_w_vals.items()}


def calc_cluster_sds(clusters_w_vals):
    return {c: np.std(vals) for c, vals in clusters_w_vals.items()}


def calc_cluster_text_dimensions(clusters_w_texts):
    cluster_dims = {}
    for c, texts in clusters_w_texts.items():
        c_top = min(t['top'] for t in texts)
        c_bottom = max(t['bottom'] for t in texts)
        c_left = min(t['left'] for t in texts)
        c_right = max(t['right'] for t in texts)
        
        cluster_dims[c] = (c_right - c_left, c_bottom - c_top)
    
    return cluster_dims


def find_best_pos_clusters(pos, num_clust_range, direction,
                           sorted_texts=None,
                           property_weights=(1, 1),
                           sds_range_thresh=float('infinity'),
                           mean_dists_range_thresh=float('infinity'),
                           min_cluster_text_height_thresh=float('-infinity'),
                           max_cluster_text_height_thresh=float('infinity'),
                           num_vals_per_clust_thresh=float('infinity')):
    """
    Assumptions:
    - y clusters should be equal spaced
    - y clusters "height" should have low SD
    (- number of items in y clusters should have low SD)
    
    - item values in x clusters should have low SD
    - number of items in x clusters should have low SD
    """
    assert direction in ('x', 'y')
    assert len(property_weights) == 2
    
    # generate different number of clusters
    fcluster_runs = []
    for n in num_clust_range:
        clust_ind = fclusterdata(pos.reshape((len(pos), 1)),  # reshape from vector to Nx1 matrix
                                 n,                     # number of clusters to find
                                 criterion='maxclust',  # stop when above n is reached
                                 metric='cityblock',    # 1D distance
                                 method='average')      # average linkage
        n_found_clust = len(np.unique(clust_ind))
        assert n_found_clust <= n
        
        if n_found_clust != n:  # it could be that we find less clusters than requested
            continue            # this is a clear sign that there're not enough elements in pos
        
        # build dicts with cluster -> vals / indices mapping
        clusters_w_vals, clusters_w_inds, clusters_w_texts = create_cluster_dicts(pos, clust_ind, sorted_texts)
        
        # calculate mean position value per cluster
        cluster_means = calc_cluster_means(clusters_w_vals)
        
        # calculate position values SD per cluster
        cluster_sds = calc_cluster_sds(clusters_w_vals)     
        
        num_vals_per_clust = [len(vals) for vals in clusters_w_vals.values()]
        # vals_per_clust_sd = np.std(num_vals_per_clust)
        vals_per_clust_range = max(num_vals_per_clust) - min(num_vals_per_clust)
        
        # calculate some properties for minimizing on them later
        if direction == 'x':
            cluster_sds_range = max(cluster_sds.values()) - min(cluster_sds.values())
            properties = (cluster_sds_range, vals_per_clust_range)
            
#            print('N=', n, 'cluster_sds_range=', cluster_sds_range, 'vals_per_clust_range=', vals_per_clust_range)
        else:  # direction == 'y'          
            sorted_clust_means = list(sorted(cluster_means.values()))
            clust_mean_dists = [c - sorted_clust_means[i-1] for i, c in enumerate(sorted_clust_means) if i > 0]
            if len(clust_mean_dists) == 1:
                # mean_dists_sd = clust_mean_dists[0]
                mean_dists_range = clust_mean_dists[0]
            else:
                # mean_dists_sd = np.std(clust_mean_dists)
                mean_dists_range = max(clust_mean_dists) - min(clust_mean_dists)
            
            if mean_dists_range > mean_dists_range_thresh:
                #print('N=', n, 'skip by mean_dists_range', mean_dists_range)
                continue
            
            cluster_text_dims = calc_cluster_text_dimensions(clusters_w_texts)
            cluster_text_heights = [dim[1] for dim in cluster_text_dims.values()]
            if min(cluster_text_heights) < min_cluster_text_height_thresh \
                    or max(cluster_text_heights) > max_cluster_text_height_thresh:
                #print('N=', n, 'skip by cluster_text_heights', min(cluster_text_heights), max(cluster_text_heights))
                continue
            #cluster_text_heights_range = max(cluster_text_heights) - min(cluster_text_heights)
            cluster_text_heights_sd = np.std(cluster_text_heights)
                        
            if vals_per_clust_range > num_vals_per_clust_thresh:
                #print('N=', n, 'skip by vals_per_clust_range', vals_per_clust_range)
                continue
            
            properties = (mean_dists_range, cluster_text_heights_sd)
        
#            print('N=', n,
#                  'mean_dists_range=', mean_dists_range,
#                  'min cluster_text_heights=', min(cluster_text_heights),
#                  'max cluster_text_heights=', max(cluster_text_heights),
#                  'cluster_text_heights_sd=', cluster_text_heights_sd,
#                  'vals_per_clust_range=', vals_per_clust_range)
        
        fcluster_runs.append((clust_ind, properties))
    
    if not len(fcluster_runs):  # no clusters found at all that met the threshold criteria
        return None
    
    n_properties = len(property_weights)
    properties_maxima = [max(x[1][p] for x in fcluster_runs) for p in range(0, n_properties)]
    
    def key_sorter(x):
        sortval = 0
        for p in range(0, n_properties):
            sortval += property_weights[p] * x[1][p] / properties_maxima[p]
        return sortval
    
    best_cluster_runs = sorted(fcluster_runs, key=key_sorter)
    
    return best_cluster_runs[0][0]

    
def put_texts_in_lines(texts):
    if not texts:
        return []
    
    mean_text_height = np.mean([t['bottom'] - t['top'] for t in texts])
    
    sorted_ts = list(sorted(texts, key=lambda x: x['top']))     # sort texts vertically
    # create list of vertical spacings between sorted texts
    text_spacings = [t['top'] - sorted_ts[i - 1]['bottom'] for i, t in enumerate(sorted_ts) if i > 0]
    text_spacings.append(0.0)   # last line
    
    # minimum positive spacing is considered to be the general vertical line spacing
    pos_sp = [v for v in text_spacings if v > 0]
    line_vspace = min(pos_sp) if pos_sp else None
    
    # go through all vertically sorted texts
    lines = []
    cur_line = []
    min_vspace_for_break = -mean_text_height / 2   # texts might overlap vertically. if the overlap is more than half
                                                   # the mean text height, it is considered a line break
    for t, spacing in zip(sorted_ts, text_spacings):
        cur_line.append(t)
        
        if spacing >= min_vspace_for_break:    # this is a line break            
            # add all texts to this line sorted by x-position
            lines.append(list(sorted(cur_line, key=lambda x: x['left'])))
            
            # add some empty line breaks if necessary
            if line_vspace:
                lines.extend([] * int(spacing / line_vspace))
            
            # reset
            cur_line = []            

    assert len(cur_line) == 0    # because last line gets a zero-spacing appended
    assert len(texts) == sum([len(l) for l in lines if len(l) > 0])     # check if all texts were put into lines
    
    return lines
    

def create_text_from_lines(lines, linebreak='\n', linejoin=' '):
    text = ''
    for l in lines:
        text += linejoin.join([t['value'] for t in l]) + linebreak
    
    return text[:-1]


def rect_from_text(t):
    return rect(t['topleft'], t['bottomright'])


def identify_sections_in_direction(texts, direction, break_section_on_distance):
    """
    Identify sections, i.e. columns or lines, in a certain direction ('x' or 'y') by breaking the input texts
    appart on <break_section_on_distance>.
    Return a list of sections, each section containing a list of texts.
    """
    # handle parameters
    direction_param_valid_values = ('x', 'y')
    if direction not in direction_param_valid_values:
        raise ValueError("direction paramater must be one of %s" % list(direction_param_valid_values))
    
    if direction == 'x':
        pos_attr = 'left'
    else:
        pos_attr = 'top'
    
    # sort by position attribute
    sorted_texts = sorted_by_attr(texts, pos_attr)
    
    # calculate the distances between the sorted texts
    dists = [t[pos_attr] - sorted_texts[i-1][pos_attr] if i > 0 else 0
             for i, t in enumerate(sorted_texts)]

    # break into sections
    texts_in_secs = []
    cur_sec = []
    for i, (t, dist) in enumerate(zip(sorted_texts, dists)):
        # if the distance is higher than break_section_on_distance
        if dist >= break_section_on_distance:
            texts_in_secs.append(cur_sec)       # save the current section
            cur_sec = []                        # and create a new section

        cur_sec.append(t)
        
    if cur_sec:  # if the last section was not added, do it now
        texts_in_secs.append(cur_sec)       # save the current section            
    
    return texts_in_secs

def merge_overlapping_sections(texts_in_secs, direction, overlap_thresh):
    """
    Merge overlapping sections of texts in <direction> 'x' or 'y' whose consecutive
    "distance" or overlap (when the distance is negative) is less than <overlap_thresh>.
    """
    direction_param_valid_values = ('x', 'y')
    if direction not in direction_param_valid_values:
        raise ValueError("direction paramater must be one of %s" % list(direction_param_valid_values))
    
    if direction == 'x':
        pos_attr = 'left'
        other_pos_attr = 'right'
    else:
        pos_attr = 'top'
        other_pos_attr = 'bottom'    
    
    # sorted section positions for left side or top side
    sec_positions1 = [sorted_by_attr(sec, pos_attr, reverse=True)[0][pos_attr] for sec in texts_in_secs]
    # sorted section positions for right side or bottom side
    sec_positions2 = [sorted_by_attr(sec, other_pos_attr, reverse=True)[0][other_pos_attr] for sec in texts_in_secs]
    
    # calculate distance/overlap between sections
    sec_positions = list(zip(sec_positions1, sec_positions2))
    sec_dists = [pos[0] - sec_positions[i-1][1] if i > 0 else 0 for i, pos in enumerate(sec_positions)]
    #print(sum([d <= 0 for d in sec_dists]))
    
    # merge sections that overlap (whose distance is less than <overlap_thresh>)
    merged_secs = []
    prev_sec = []
    for i, dist in enumerate(sec_dists):
        cur_sec = texts_in_secs[i]
        if dist < overlap_thresh:
            sec = cur_sec + prev_sec
            if len(merged_secs) > 0:
                merged_secs.pop()
        else:
            sec = cur_sec
        
        merged_secs.append(sec)
        prev_sec = sec
    
    assert len(flatten_list(texts_in_secs)) == len(flatten_list(merged_secs))
    
    return merged_secs


def merge_small_sections(texts_in_secs, min_num_texts):
    """
    Merge sections that are too small, i.e. have too few "content" which means that their number
    of texts is lower than or equal <min_num_texts>.
    """
    merged_secs = []
    prev_sec = None
    for cur_sec in texts_in_secs:
        if prev_sec:
            if len(cur_sec) <= min_num_texts:  # number of texts is too low
                sec = cur_sec + prev_sec       # merge this section with the previous section
                if len(merged_secs) > 0:       # remove the prev. section from the final list
                    merged_secs.pop()          # in order to add the merged section later
            else:
                sec = cur_sec
        else:
            sec = cur_sec
        
        merged_secs.append(sec)   # add the (possibly merged) section
        prev_sec = sec

    assert len(flatten_list(texts_in_secs)) == len(flatten_list(merged_secs))

    return merged_secs
