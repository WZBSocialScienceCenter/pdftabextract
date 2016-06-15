# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:40:39 2016

@author: mkonrad
"""

from collections import defaultdict, OrderedDict
from logging import warning

import re

import json
import csv

from scipy.cluster.hierarchy import fclusterdata
import numpy as np

from geom import pt, rect, rectintersect
from common import read_xml, parse_pages, get_bodytexts, divide_texts_horizontally, sorted_by_attr, \
                   texts_at_page_corners


HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5


# TODO: test with subpages that are not really subpages
# TODO: do not hardcode configuration settings
# TODO: real logging instead of print()

#%%
def cond_topleft_text(t):
    text = t['value'].strip()
    return re.match(r'^\d+', text) is not None
    
def cond_bottomleft_text(t):
    text = t['value'].strip()
    return re.search(r'^(G|WS)$', text) is not None

corner_box_condition_fns = (cond_topleft_text, None, None, cond_bottomleft_text)

#%%
def extract_tabular_data_from_pdf2xml_file(xmlfile, corner_box_cond_fns=None):
    # get subpages (if there're two pages on a single scanned page)
    subpages = get_subpages_from_xml(xmlfile)
    
    # analyze the row/column layout of each page and return these layouts,
    # a list of invalid subpages (no tabular layout could be recognized),
    # and a list of common column positions
    layouts, invalid_layouts, col_positions, mean_row_height = analyze_subpage_layouts(subpages, corner_box_cond_fns)
    
    output = OrderedDict()
    # go through all subpages
    for p_id in sorted(subpages.keys(), key=lambda x: x[0]):
        sub_p = subpages[p_id]            
        
        pagenum, pageside = p_id
        
        if pagenum not in output:
            output[pagenum] = OrderedDict()
        
        # get the column positions
        subp_col_positions = list(np.array(col_positions) + sub_p['x_offset'])
            
        if p_id in invalid_layouts:
            row_positions = guess_row_positions(sub_p, mean_row_height, subp_col_positions)
            if row_positions:
                print("subpage %d/%s layout: guessed %d rows (mean row height %f)"
                      % (sub_p['number'], sub_p['subpage'], len(row_positions), mean_row_height))
        else:        
            # get the row positions
            row_positions = layouts[p_id][0]
        
        if not row_positions:
            print("subpage %d/%s layout: no row positions identified -- skipped" % (sub_p['number'], sub_p['subpage']))
            continue
        
        # fit the textboxes from this subpage into the tabular grid defined by the
        # row and column positions
        table = create_datatable_from_subpage(sub_p,
                                              row_positions=row_positions,
                                              col_positions=subp_col_positions)

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

    return output


def save_tabular_data_dict_as_json(data, jsonfile):
    with open(jsonfile, 'w') as f:
        json.dump(data, f, indent=2)


def save_tabular_data_dict_as_csv(data, csvfile):
    with open(csvfile, 'w') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        csv_rows = []
        for p_num, subpages in data.items():
            for p_side, table_texts in subpages.items():
                for table_row in table_texts:
                    csv_rows.append([p_num, p_side] + table_row)
        
        n_addcols = len(csv_rows[0]) - 2
        csv_header = ['page_num', 'page_side'] + ['col' + str(i+1) for i in range(n_addcols)]
        
        csvwriter.writerows([csv_header] + csv_rows)


def get_subpages_from_xml(xmlfile):
    tree, root = read_xml(xmlfile)
    
    pages = parse_pages(root)

    subpages = {}
    
    for p_num, page in pages.items():
        # strip off footer and header
        bodytexts = get_bodytexts(page, HEADER_RATIO, FOOTER_RATIO)
        
        if DIVIDE_RATIO:
            page_subpages = divide_texts_horizontally(page, DIVIDE_RATIO, bodytexts)
        else:
            page_subpages = (page, )
            
        for sub_p in page_subpages:
            p_id = (sub_p['number'], sub_p['subpage'])        
            subpages[p_id] = sub_p
    
    return subpages


def guess_row_positions(subpage, mean_row_height, col_positions):
    if not len(col_positions) > 1:
        raise ValueError('number of detected columns must be at least 2')
    
    texts_by_y = sorted_by_attr(subpage['texts'], 'top')
    
    # borders of the first column
    first_col_left, first_col_right = col_positions[0:2]
    
    # find the first (top-most) text that completely fits in a possible table cell in the first column
    top_text = None
    for t in texts_by_y:
        t_rect = rect_from_text(t)
        cell_rect = rect(pt(first_col_left, t['top']), pt(first_col_right, t['top'] + mean_row_height))        
        isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
        if isect == 1.0:
            top_text = t
            break
        
    if not top_text:
        warning("subpage %d/%s: could not find top text" % (subpage['number'], subpage['subpage']))
        return None
    
    # borders of the last column
    #last_col_left, last_col_right = col_positions[-2:]
    
    # find the last (lowest) text that completely fits in a possible table cell in the first column
    bottom_text = None
    for t in reversed(texts_by_y):
        t_rect = rect_from_text(t)
        cell_rect = rect(pt(first_col_left, t['top']), pt(first_col_right, t['top'] + mean_row_height))
        isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
        if isect == 1.0:
            bottom_text = t
            break
    
    if not bottom_text:
        warning("subpage %d/%s: could not find bottom text" % (subpage['number'], subpage['subpage']))
        return None
    
    top_border = int(np.round(top_text['top']))
    bottom_border = int(np.round(bottom_text['top'] + mean_row_height))
    
    table_height = bottom_border - top_border
    n_rows, remainder = divmod(table_height, mean_row_height)
    if remainder / mean_row_height > 0.5:   # seems like the number of rows doesn't really fit
        warning("subpage %d/%s: the number of rows doesn't really fit the guessed table height"
                % (subpage['number'], subpage['subpage']))
        return None                         # we assume this is an invalid table layout
    else:
        optimal_row_height = table_height // n_rows
        return list(range(top_border, bottom_border, optimal_row_height))[:n_rows]


def analyze_subpage_layouts(subpages, corner_box_cond_fns=None):
    # find the column and row borders for each subpage
    layouts = {}
    for p_id, sub_p in subpages.items():        
        try:
            col_positions, row_positions = find_col_and_row_positions_in_subpage(sub_p, corner_box_cond_fns)
        except ValueError as e:
            print("subpage %d/%s layout: skipped ('%s')" % (sub_p['number'], sub_p['subpage'], str(e)))
            col_positions, row_positions = None, None
        
        if col_positions and row_positions:
            n_rows = len(row_positions)
            n_cols = len(col_positions)
            
            print("subpage %d/%s layout: %d cols, %d rows" % (sub_p['number'], sub_p['subpage'], n_cols, n_rows))
            
            layout = (row_positions, col_positions)
        else:
            print("subpage %d/%s layout: invalid column/row positions: %s/%s"
                  % (sub_p['number'], sub_p['subpage'], col_positions, row_positions))
            layout = None
        
        layouts[p_id] = layout
    
    # get the row and column positions of all valid subpages
    all_row_pos, all_col_pos = zip(*[(np.array(layout[0]), np.array(layout[1]) - subpages[p_id]['x_offset'])
                               for p_id, layout in layouts.items()
                               if layout is not None])
    
    # get all numbers of rows and columns across the subpages
    nrows = [len(row_pos) for row_pos in all_row_pos]
    ncols = [len(col_pos) for col_pos in all_col_pos]
    
    nrows_median = np.median(nrows)
    ncols_median = np.median(ncols)
    
    print("median number of rows:", nrows_median)
    print("median number of columns:", ncols_median)
    
    row_pos_w_median_len = [row_pos for row_pos in all_row_pos if len(row_pos) == nrows_median]
    selected_row_pos = [row_pos[2:int(nrows_median)-1] for row_pos in row_pos_w_median_len]
    row_height_means = []
    for row_pos in selected_row_pos:
        row_height_means.append(np.mean([pos - row_pos[i-1] for i, pos in enumerate(row_pos[1:])]))
    overall_row_height_mean = int(np.round(np.mean(row_height_means)))
    
    # find the "best" (median) column positions for the median number of columns
    col_pos_w_median_len = [col_pos for col_pos in all_col_pos if len(col_pos) == ncols_median]
    best_col_pos = [list() for _ in range(int(ncols_median))]
    for col_positions in col_pos_w_median_len:
        for i, pos in enumerate(col_positions):
            best_col_pos[i].append(pos)
    
    best_col_pos_medians = [np.median(best_col_pos[i]) for i in range(int(ncols_median))]

    # get list of subpages without proper layout    
    invalid_layouts = [p_id for p_id, layout in layouts.items() if layout is None]
    
    return layouts, invalid_layouts, best_col_pos_medians, overall_row_height_mean


def create_datatable_from_subpage(subpage, row_positions, col_positions):    
    n_rows = len(row_positions)
    n_cols = len(col_positions)

    row_ranges = position_ranges(row_positions, subpage['height'])
    col_ranges = position_ranges(col_positions, subpage['x_offset'] + subpage['width'])
    
    # create a grid with rectangles of table cells
    grid = {(r_i, c_i): rect(pt(l, t), pt(r, b)) for r_i, (t, b) in enumerate(row_ranges)
                                                 for c_i, (l, r) in enumerate(col_ranges)}
    
    # create an empty table with the found dimensions
    # each cell will have a list with textblocks inside
    table = np.empty((n_rows, n_cols), dtype='object')
    # np.full does not work with with list as fill value so we have to do it like this:
    for j in range(n_rows):
        for k in range(n_cols):
            table[j, k] = []
    
    # iterate through the textblocks of this page
    for t in subpage['texts']:
        t_rect = rect_from_text(t)   # rectangle of the textbox
        
        # find out the cells with which this textbox rectangle intersects
        cell_isects = []
        for idx, cell_rect in grid.items():
            isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
            if isect is not None and isect > 0:
                cell_isects.append((idx, isect))
        
        if len(cell_isects) > 0:
            # find out the cell with most overlap
            max_isect_val = max([x[1] for x in cell_isects])
            if max_isect_val < 0.5:
                warning("subpage %d/%s: low best cell intersection value: %f" % (subpage['number'], subpage['subpage'], max_isect_val))
            best_isects = [x for x in cell_isects if x[1] == max_isect_val]
            if len(best_isects) > 1:
                warning("subpage %d/%s: multiple (%d) best cell intersections" % (subpage['number'], subpage['subpage'], len(best_isects)))
            best_idx = best_isects[0][0]
            
            # add this textblock to the table at the found cell index
            table[best_idx].append(t)
        else:
            warning("subpage %d/%s: no cell found for textblock '%s'" % (subpage['number'], subpage['subpage'], t))
    
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
        
    x_clust_ind = find_best_pos_clusters(xs, range(3, 9), 'x',
                                         property_weights=(1, 5))
    if xs is None or x_clust_ind is None:
        col_positions = None
    else:
        x_clust_w_vals, _, _ = create_cluster_dicts(xs, x_clust_ind)
        x_clust_w_vals = {c: vals for c, vals in x_clust_w_vals.items() if len(vals) > 3}
        col_positions = positions_list_from_clustervalues(x_clust_w_vals.values())
    
    y_clust_ind = find_best_pos_clusters(ys, range(2, 15), 'y',
                                         sorted_texts=texts_by_y,
                                         property_weights=(1, 1),
                                         min_cluster_text_height_thresh=25,
                                         max_cluster_text_height_thresh=70,
                                         mean_dists_range_thresh=30)
    if y_clust_ind is None:
        row_positions = None
    else:
        y_clust_w_vals, _, _ = create_cluster_dicts(ys, y_clust_ind)
        row_positions = positions_list_from_clustervalues(y_clust_w_vals.values())
    
    return col_positions, row_positions


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
                continue
            
            cluster_text_dims = calc_cluster_text_dimensions(clusters_w_texts)
            cluster_text_heights = [dim[1] for dim in cluster_text_dims.values()]
            print('min cluster_text_heights=', min(cluster_text_heights))
            if min(cluster_text_heights) < min_cluster_text_height_thresh \
                    or max(cluster_text_heights) > max_cluster_text_height_thresh:
                continue
            #cluster_text_heights_range = max(cluster_text_heights) - min(cluster_text_heights)
            cluster_text_heights_sd = np.std(cluster_text_heights)
                        
            if vals_per_clust_range > num_vals_per_clust_thresh:
                continue
            
            properties = (mean_dists_range, cluster_text_heights_sd)
        
#            print('N=', n,
#                  'mean_dists_range=', mean_dists_range,
#                  'cluster_text_heights_range=', cluster_text_heights_range,
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
    for t, spacing in zip(sorted_ts, text_spacings):
        cur_line.append(t)
        
        if spacing >= 0:    # this is a line break            
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