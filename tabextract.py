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

from geom import pt, rect, rectintersect
from common import read_xml, parse_pages, get_bodytexts, divide_texts_horizontally


HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5


# TODO: test with right subpages
# TODO: test with subpages that are not really subpages
# TODO: do not hardcode configuration settings


#%%

def extract_tabular_data_from_pdf2xml_file(xmlfile):
    subpages = get_subpages_from_xml(xmlfile)
    
    layouts, invalid_subpages, col_positions = analyze_subpage_layouts(subpages)
    
    output = OrderedDict()
    for p_id in sorted(subpages.keys(), key=lambda x: x[0]):
        sub_p = subpages[p_id]
        if p_id in invalid_subpages:
            print("subpage %d/%s layout: skipped" % (sub_p['number'], sub_p['subpage']))
            continue
        
        pagenum, pageside = p_id
        
        if pagenum not in output:
            output[pagenum] = OrderedDict()
        
        row_positions = layouts[p_id][0]
        subp_col_positions = list(np.array(col_positions) + sub_p['x_offset'])
        table = create_datatable_from_subpage(sub_p, row_positions=row_positions, col_positions=subp_col_positions)
        table_texts = []
        for row in table:
            row_texts = []
            for cell in row:
                cell_lines = put_texts_in_lines(cell)
                row_texts.append(create_text_from_lines(cell_lines))
            table_texts.append(row_texts)
        
        output[pagenum][pageside] = table_texts

    return output


def save_tabular_data_dict_as_json(data, jsonfile):
    with open(jsonfile, 'w') as f:
        json.dump(data, f, indent=2)


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


def analyze_subpage_layouts(subpages):
    # find the column and row borders for each subpage
    layouts = {}
    for p_id, sub_p in subpages.items():        
        try:
            col_positions, row_positions = find_col_and_row_positions_in_subpage(sub_p)
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
    nrows = [len(row_pos) for row_pos in all_row_pos]   # we don't actually need this
    ncols = [len(col_pos) for col_pos in all_col_pos]
    
    nrows_median = np.median(nrows)     # we don't actually need this
    ncols_median = np.median(ncols)
    
    print("median number of rows:", nrows_median)
    print("median number of columns:", ncols_median)
    
    # find the "best" (median) column positions for the median number of columns
    col_pos_w_median_len = [col_pos for col_pos in all_col_pos if len(col_pos) == ncols_median]
    best_col_pos = [list() for _ in range(int(ncols_median))]
    for col_positions in col_pos_w_median_len:
        for i, pos in enumerate(col_positions):
            best_col_pos[i].append(pos)
    
    best_col_pos_medians = [np.median(best_col_pos[i]) for i in range(int(ncols_median))]

    # get list of "invalid" subpages / subpages without proper layout    
    invalid_subpages = [p_id for p_id, layout in layouts.items() if layout is None]
    
    return layouts, invalid_subpages, best_col_pos_medians


def table_debugprint(table):
    textmat = np.empty(table.shape, dtype='object')
    
    for j in range(table.shape[0]):
        for k in range(table.shape[1]):
            texts = [t['value'] for t in table[j, k]]
            textmat[j, k] = ', '.join(texts)

    print(textmat)


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
        t_rect = rect(t['topleft'], t['bottomright'])   # rectangle of the textbox
        
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
    

def find_col_and_row_positions_in_subpage(subpage):
    if len(subpage['texts']) < 3:
        raise ValueError('insufficient number of texts on subpage %d/%s' % (subpage['number'], subpage['subpage']))
    
    xs = []
    ys = []
    
    for t in subpage['texts']:
        xs.append(t['left'])
        ys.append(t['top'])
    
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    
    xs_arr, x_clust_ind = find_best_pos_clusters(xs_arr, range(3, 9), 'x', property_weights=(1, 5))
    if xs_arr is None or x_clust_ind is None:
        col_positions = None
    else:
        x_clust_w_vals, _ = create_cluster_dicts(xs_arr, x_clust_ind)
        x_clust_w_vals = {c: vals for c, vals in x_clust_w_vals.items() if len(vals) > 3}
        col_positions = positions_list_from_clustervalues(x_clust_w_vals.values())
    
    ys_arr, y_clust_ind = find_best_pos_clusters(ys_arr, range(2, 15), 'y', mean_dists_range_thresh=30)
    if ys_arr is None or y_clust_ind is None:
        row_positions = None
    else:
        y_clust_w_vals, _ = create_cluster_dicts(ys_arr, y_clust_ind)
        row_positions = positions_list_from_clustervalues(y_clust_w_vals.values())
    
    return col_positions, row_positions


def position_ranges(positions, last_val):
    p = positions + [last_val]
    return [(p[i-1], v) for i, v in enumerate(p) if i > 0]


def positions_list_from_clustervalues(clust_vals, val_filter=min):
    positions = [val_filter(vals) for vals in clust_vals]
    
    return list(sorted(positions))


def create_cluster_dicts(vals, clust_ind):
    # build dicts with ...        
    clusters_w_vals = defaultdict(list)     # ... cluster -> [values] mapping
    clusters_w_inds = defaultdict(list)     # ... cluster -> [indices] mapping
    for i, (v, c) in enumerate(zip(vals, clust_ind)):
        clusters_w_vals[c].append(v)
        clusters_w_inds[c].append(i)
    
    return clusters_w_vals, clusters_w_inds


def calc_cluster_means(clusters_w_vals):
    # calculate mean position value per cluster
    return {c: np.mean(vals) for c, vals in clusters_w_vals.items()}


def calc_cluster_sds(clusters_w_vals):
    return {c: np.std(vals) for c, vals in clusters_w_vals.items()}


def find_best_pos_clusters(pos, num_clust_range, direction,
                           property_weights=(1, 1),
                           sds_range_thresh=float('infinity'),
                           mean_dists_range_thresh=float('infinity'),
                           num_vals_per_clust_thresh=float('infinity')):
    """
    Assumptions:
    - y clusters should be equal spaced
    - number of items in y clusters should be equal distributed
    
    - items in x clusters should have low standard deviation
    - number of items in x clusters should be equal distributed
    """
    assert direction in ('x', 'y')
    assert len(property_weights) == 2
    
    # sort input positions first
    pos.sort()
    
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
        clusters_w_vals, clusters_w_inds = create_cluster_dicts(pos, clust_ind)
        
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
            
            print('N=', n, 'cluster_sds_range=', cluster_sds_range, 'vals_per_clust_range=', vals_per_clust_range)
        else:            
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
                        
            if vals_per_clust_range > num_vals_per_clust_thresh:
                continue
            
            properties = (mean_dists_range, vals_per_clust_range)
        
            print('N=', n,
                  # 'dists SD=', mean_dists_sd,
                  'dists range=', mean_dists_range,
                  # 'num. values SD=', vals_per_clust_sd,
                  'num. values range=', vals_per_clust_range)
        
        fcluster_runs.append((clust_ind, properties))
    
    if not len(fcluster_runs):  # no clusters found at all that met the threshold criteria
        return None, None
    
    n_properties = len(property_weights)
    properties_maxima = [max(x[1][p] for x in fcluster_runs) for p in range(0, n_properties)]
    
    def key_sorter(x):
        sortval = 0
        for p in range(0, n_properties):
            sortval += property_weights[p] * x[1][p] / properties_maxima[p]
        return sortval
    
    best_cluster_runs = sorted(fcluster_runs, key=key_sorter)
    
    return pos, best_cluster_runs[0][0]

    
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
