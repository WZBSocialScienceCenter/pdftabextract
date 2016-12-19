# -*- coding: utf-8 -*-
"""
Common clustering functions and utilities.

Created on Fri Dec 16 14:14:30 2016

@author: mkonrad
"""

import numpy as np


def find_clusters_1d_break_dist(vals, dist_thresh):
    """
    Very simple clusting in 1D: Sort <vals> and calculate distance between values. Form clusters when <dist_thresh> is
    exceeded.
    
    Returns a list if clusters, where each element in the list is a np.array with indices of <vals>.
    """
    if type(vals) is not np.ndarray:
        raise ValueError("vals must be a NumPy array")
    
    clusters = []
    
    if len(vals) > 0:
        pos_indices_sorted = np.argsort(vals)      # indices of sorted values
        gaps = np.diff(vals[pos_indices_sorted])   # calculate distance between sorted values
        
        cur_clust = [pos_indices_sorted[0]]  # initialize with first index
        
        if len(vals) > 1:
            for idx, gap in zip(pos_indices_sorted[1:], gaps):
                if gap >= dist_thresh:           # create new cluster
                    clusters.append(np.array(cur_clust))
                    cur_clust = []
                cur_clust.append(idx)
            
        clusters.append(np.array(cur_clust))
    
    assert len(vals) == sum(map(len, clusters))
    
    return clusters

#%% Helper functions
    
def zip_clusters_and_values(clusters, values):
    clusters_w_vals = []
    for c_ind in clusters:
        c_vals = values[c_ind]
        clusters_w_vals.append((c_ind, c_vals))
    
    return clusters_w_vals


def calc_cluster_centers_1d(clusters_w_vals, method=np.median):
    return [method(vals) for _, vals in clusters_w_vals]

def calc_cluster_centers_range(clusters_w_vals, reduce_clusters_method=np.median, return_centers=False):
    centers = calc_cluster_centers_1d(clusters_w_vals, method=reduce_clusters_method)
    rng = max(centers) - min(centers)
    if return_centers:
        return rng, centers
    else:
        return rng


