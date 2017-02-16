# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:51:46 2017

@author: mkonrad
"""

import pytest
from hypothesis import given
import hypothesis.strategies as st 
import numpy as np

from pdftabextract.clustering import find_clusters_1d_break_dist, zip_clusters_and_values


@given(st.lists(st.integers(min_value=-10000, max_value=10000)),
       st.integers(min_value=-10000, max_value=10000))
def test_find_clusters_1d_break_dist(seq, delta):
    with pytest.raises(TypeError):  # first param must be np.array
        find_clusters_1d_break_dist(seq, delta)
    
    arr = np.array(seq)
    
    if delta < 0:
        with pytest.raises(ValueError):   # delta must be >= 0
            find_clusters_1d_break_dist(arr, delta)
        return
    
    clusts = find_clusters_1d_break_dist(arr, delta)

    # types and return length must match
    assert type(clusts) is list
    assert sum(map(len, clusts)) == len(seq)
    
    idx_list = []
    for c in clusts:
        idx_list.extend(c)
    
    assert len(idx_list) == len(seq)
    recon = arr[idx_list]
    recon_sorted = np.sort(recon)
    seq_sorted = np.sort(seq)
    
    # values in clusters and in input must match
    assert np.array_equal(recon_sorted, seq_sorted)
    
    if len(seq) > 1:
        clust_borders = []
        for c in clusts:
            v = arr[c]
            
            # inside clusters, the gaps must be < delta
            if len(v) > 1:
                max_dist_in_clust = max(np.diff(np.sort(v)))
                assert max_dist_in_clust < delta
            
            v_min = np.min(v)
            v_max = np.max(v)
            clust_borders.append((v_min, v_max))
        
        clust_borders = sorted(clust_borders, key=lambda x: x[0])
        
        if len(clusts) > 1:
            # between the clusters, the gaps must be >= delta           
            gaps = []
            prev_max = None
            for v_min, v_max in clust_borders:
                if prev_max is not None:
                    gaps.append(v_min - prev_max)
                prev_max = v_max
            
            assert min(gaps) >= delta

@given(st.lists(st.integers(min_value=-10000, max_value=10000)),
       st.integers(min_value=-10000, max_value=10000))
def test_zip_clusters_and_values(seq, delta):
    arr = np.array(seq)
    
    try:
        clusts = find_clusters_1d_break_dist(arr, delta)
    except:   # exceptions are tested in test_find_clusters_1d_break_dist
        return
    
    with pytest.raises(TypeError):  # second param must be np.array
        zip_clusters_and_values(clusts, seq)
    
    clusts_w_vals = zip_clusters_and_values(clusts, arr)
    assert len(clusts_w_vals) == len(clusts)
    
    for tup in clusts_w_vals:
        assert len(tup) == 2
        ind, vals = tup
        assert len(ind) > 0
        assert len(ind) == len(vals)
        assert np.array_equal(arr[ind], vals)

