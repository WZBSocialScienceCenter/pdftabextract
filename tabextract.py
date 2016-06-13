# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:40:39 2016

@author: mkonrad
"""

from collections import defaultdict
from logging import warning

from scipy.cluster.hierarchy import fclusterdata
import numpy as np


from geom import pt, rect, rectintersect


# TODO: test with right subpages
# TODO: test with subpages that are not really subpages
# TODO: do not hardcode configuration settings


#%%


def analyze_subpage_layouts(subpages):
    layouts = {}
    for p_id, sub_p in subpages.items():
        print(sub_p['number'], sub_p['subpage'])
        
        # find the column and row borders
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
    
    return layouts


def table_debugprint(table):
    textmat = np.empty(table.shape, dtype='object')
    
    for j in range(table.shape[0]):
        for k in range(table.shape[1]):
            texts = [t['value'] for t in table[j, k]]
            textmat[j, k] = ', '.join(texts)

    print(textmat)


def create_datatable_from_subpage(subpage):
    # find the column and row borders
    col_positions, row_positions = find_col_and_row_positions_in_subpage(subpage)
    
    n_rows = len(row_positions)
    n_cols = len(col_positions)
    
    print("subpage %d/%s layout: %d cols, %d rows" % (subpage['number'], subpage['subpage'], n_cols, n_rows))
    
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
    textblocks = subpage['texts'][:]
    for t in textblocks[:]:
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
            textblocks.remove(t)    # remove it from the textblocks list
    
    # now all textblocks that are left could not be fitted into a cell
    # report that here
    for t in textblocks:
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
    sorted_ts = list(sorted(texts, key=lambda x: x['top']))
    text_spacings = [t['top'] - sorted_ts[i - 1]['bottom'] for i, t in enumerate(sorted_ts) if i > 0]
    text_spacings.append(0.0)   # last line
    
    pos_text_spacings = [v for v in text_spacings if v > 0]    
    line_vspace = min(pos_text_spacings)
    
    lines = []
    cur_line = []
    for t, spacing in zip(sorted_ts, text_spacings):
        cur_line.append(t)
        
        if spacing >= 0:    # this is a line break            
            # add all texts to this line sorted by x-position
            lines.append(list(sorted(cur_line, key=lambda x: x['left'])))
            
            # add some empty line breaks if necessary
            lines.extend([] * int(spacing / line_vspace))
            cur_line = []            

    assert len(cur_line) == 0    # because last line gets a zero-spacing appended
    assert len(texts) == sum([len(l) for l in lines if len(l) > 0])     # check if all texts were put into lines
    
    return lines
    

def create_text_from_lines(lines, linebreak='\n', linejoin=' '):
    text = ''
    for l in lines:
        text += linejoin.join([t['value'] for t in l]) + linebreak
    
    return text
