# -*- coding: utf-8 -*-
"""
Common clustering functions and utilities.

Created on Fri Dec 16 14:14:30 2016

@author: mkonrad
"""

import itertools

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

        
def array_match_difference_1d(a, b):
    """Return the summed difference between the elements in a and b."""
    if len(a) != len(b):
        raise ValueError('Both arrays must have the same length')
    
    if len(a) == 0:
        raise ValueError('Arrays must be filled')
    
    if type(a) is not np.ndarray:
        a = np.array(a)
    if type(b) is not np.ndarray:
        b = np.array(b)
    
    return np.sum(np.abs(a - b))
    

def find_best_matching_array(base_arr, model_arr):
    """
    Find an array <best_arr> based on <base_arr> that fits best to <model_arr> if their sizes differ.
    <best_arr> will have the same size as <model_arr> and either has surplus elements removed (if <base_arr> is
    bigger than <model_arr>) or missing elements added from <model_arr> (if <base_arr> is smaller than <model_arr>).
    
    Returns the best fitting array and the summed difference of this array and <model_arr>.
    
    It uses a brute force method so this is slow for big arrays.
    
    Example:
        
    values = [
        [0,  10,     30,         40],
        [0,  11,     29,         42],
        [10, 21, 25, 39,         52],
        [0,   9, 15, 29, 32,     41],
        [0,  10,     29, 35, 36, 40],
        [0,   9,                 41],
        [0,          33,           ],
    ]
    
    model = np.array(values[0])  # first row is the "model" -> we know that this is correct
    for row in values[1:]:
        row = np.array(row)
        print(row)
        corrected_row, diffsum = find_best_matching_array(row, model)
        print(corrected_row)
        print(diffsum)
        print()

    Output:
        [ 0 11 29 42]
        [ 0 11 29 42]
        4
        
        [10 21 25 39 52]
        [10 21 39 52]
        4
        
        [ 0  9 15 29 32 41]
        [ 0  9 29 41]
        3
        
        [ 0 10 29 35 36 40]
        [ 0 10 29 40]
        1
        
        [ 0  9 41]
        [ 0  9 30 41]
        2
        
        [ 0 33]
        [ 0 10 33 40]
        3
    """
    if type(base_arr) is not np.ndarray:
        raise ValueError("base_arr must be NumPy array")
    if type(model_arr) is not np.ndarray:
        raise ValueError("model_arr must be NumPy array")
    
    amount_diff = len(base_arr)  - len(model_arr)
    
    if amount_diff > 0:    # too many values in base_arr
        # go through all possible combinations of surplus elements in the base_arr and
        # measure the match difference and save it to "candidates"
        del_indices_combi = itertools.combinations(range(len(base_arr)), amount_diff)
        candidates = []
        for del_ind in del_indices_combi:
            candidate_arr = np.delete(base_arr, del_ind)
            # model_arr is normalized -> add first value as offset
            center_medians_w_offset = model_arr + candidate_arr[0]
            diff = array_match_difference_1d(candidate_arr, center_medians_w_offset)
            candidates.append((candidate_arr, diff))
        
        best_arr, diff = sorted(candidates, key=lambda x: x[1])[0]
    elif amount_diff < 0:  # too few values in base_arr
        # this time, reduce the model_arr so that it fits the number of values in base_arr
        # i.e. we find the best candidate of all adjusted model_arr first
        del_indices_combi = itertools.combinations(range(len(model_arr)), -amount_diff)
        candidates = []
        for del_ind in del_indices_combi:
            candidate_arr = np.delete(model_arr, del_ind)
            # model_arr is normalized -> add first value as offset
            diff = array_match_difference_1d(candidate_arr + base_arr[0], base_arr)
            candidates.append((del_ind, diff))
            
        add_ind, _ = sorted(candidates, key=lambda x: x[1])[0]
        model_arr_w_offset = model_arr + base_arr[0]
        
        # take the missing values from best_model_arr
        best_arr = fill_array_a_with_values_from_b(base_arr, model_arr_w_offset, add_ind)
        diff = array_match_difference_1d(best_arr, model_arr_w_offset)
    else:                  # number of values matches
        diff = array_match_difference_1d(base_arr, model_arr + base_arr[0])
        best_arr = base_arr
    
    return best_arr, diff


