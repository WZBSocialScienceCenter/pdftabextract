# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:40:39 2016

@author: mkonrad
"""

from collections import defaultdict

from scipy.cluster.hierarchy import fclusterdata
import numpy as np


# TODO: test with right subpages
# TODO: test with subpages that are not really subpages
# TODO: do not hardcode configuration settings


#%%


def matrix_of_text_data_from_subpage(subpage):
    text_objs, (n_rows, n_cols) = organize_textobjs_in_table_cells(subpage)
    
    textmat = np.empty((n_rows, n_cols), dtype='object')
    
    for row in range(n_rows):
        for col in range(n_cols):
            texts = [t['value'] for t in text_objs[row][col]]
            textmat[row, col] = ', '.join(texts)
    
    return textmat


def organize_textobjs_in_table_cells(subpage):
    col_positions, row_positions = find_col_and_row_positions_in_subpage(subpage)
    n_rows = len(row_positions)
    n_cols = len(col_positions)
    print("subpage %d/%s layout: %d cols, %d rows" % (subpage['number'], subpage['subpage'], n_cols, n_rows))

    row_ranges = position_ranges(row_positions, subpage['height'])
    texts_in_rows = []
    print('create rows')
    for row_rng in row_ranges:
        texts = []
        for t in subpage['texts']:
            if row_rng[0] <= t['top'] < row_rng[1]:
                texts.append(t)
        texts_in_rows.append(texts)
    n_texts_in_rows = sum(len(texts) for texts in texts_in_rows)
    
    assert len(row_ranges) == n_rows
    assert len(subpage['texts']) == n_texts_in_rows
    
    col_ranges = position_ranges(col_positions, subpage['x_offset'] + subpage['width'])
    text_data = []
    for row_texts in texts_in_rows:
        row_texts_ordered = []
        for col_rng in col_ranges:
            cell_texts = []
            for t in row_texts:
                if col_rng[0] - 30 <= t['left'] < col_rng[1]:
                    cell_texts.append(t)
            row_texts_ordered.append(cell_texts)
        text_data.append(row_texts_ordered)
    
    n_texts_in_cells = sum((sum(len(texts) for texts in r) for r in text_data))
    assert len(subpage['texts']) == n_texts_in_cells
    
    return text_data, (n_rows, n_cols)
    

def find_col_and_row_positions_in_subpage(subpage):
    xs = []
    ys = []
    
    for t in subpage['texts']:
        xs.append(t['left'])
        ys.append(t['top'])
    
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    
    xs_arr, x_clust_ind = find_best_pos_clusters(xs_arr, range(3, 9), 'x', property_weights=(1, 5))
    x_clust_w_vals, _ = create_cluster_dicts(xs_arr, x_clust_ind)
    x_clust_w_vals = {c: vals for c, vals in x_clust_w_vals.items() if len(vals) > 3}
    
    col_positions = positions_list_from_clustervalues(x_clust_w_vals.values())
    
    ys_arr, y_clust_ind = find_best_pos_clusters(ys_arr, range(2, 15), 'y', mean_dists_range_thresh=30)
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
        return None
    
    n_properties = len(property_weights)
    properties_maxima = [max(x[1][p] for x in fcluster_runs) for p in range(0, n_properties)]
    
    def key_sorter(x):
        sortval = 0
        for p in range(0, n_properties):
            sortval += property_weights[p] * x[1][p] / properties_maxima[p]
        return sortval
    
    best_cluster_runs = sorted(fcluster_runs, key=key_sorter)
    
    return pos, best_cluster_runs[0][0]
    