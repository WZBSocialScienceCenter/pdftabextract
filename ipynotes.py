# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:11:33 2016

@author: mkonrad
"""

import fixrotation


#%%

def cond_topleft_text(t):
    return t['value'].strip() == 'G'
cond_bottomleft_text = cond_topleft_text

#def cond_topright_text(t):
#    m = re.search(r'^\d{2}$', t['value'].strip())
#    w = t['width']
#    h = t['height']
#    return m and abs(15 - w) <= 3 and abs(12 - h) <= 2

def cond_topright_text(t):
    return False
cond_bottomright_text = cond_topright_text

corner_box_cond_fns = (cond_topleft_text, cond_topright_text, cond_bottomright_text, cond_bottomleft_text)
fixrotation.fix_rotation('testxmls/1992_93.pdf.xml', 'testxmls/1992_93_rotback.pdf.xml', corner_box_cond_fns)

#%%

from common import read_xml, parse_pages, get_bodytexts, divide_texts_horizontally

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5
# DIVIDE_RATIO = None

LEFTMOST_COL_ALIGN = 'topleft'     # topleft, topright or center
RIGHTMOST_COL_ALIGN = 'topleft'    # topleft, topright or center

MIN_CONTENTLENGTH_MEAN_DEV_RATIO = 0.2

tree, root = read_xml('testxmls/1992_93_rotback.pdf.xml')

# get pages objects    
pages = parse_pages(root)
    
pages_bodytexts = {}
pages_contentlengths = {}
subpages = {}
for p_num, page in pages.items():
    # strip off footer and header
    bodytexts = get_bodytexts(page, HEADER_RATIO, FOOTER_RATIO)
    
    if DIVIDE_RATIO:
        page_subpages = divide_texts_horizontally(page, DIVIDE_RATIO, bodytexts)
    else:
        page_subpages = (page, )
    
    for sub_p in page_subpages:
        if 'subpage' in sub_p:
            p_id = (sub_p['number'], sub_p['subpage'])
        else:
            p_id = (sub_p['number'], )
        pages_bodytexts[p_id] = sub_p['texts']
        contentlength = sum([len(t['value']) for t in sub_p['texts']])
        pages_contentlengths[p_id] = contentlength
    
        subpages[p_id] = sub_p
mean_contentlength = sum([length for length in pages_contentlengths.values()]) / len(pages_contentlengths)

#%%
import matplotlib.pyplot as plt
import numpy as np

subpage = subpages[(17, 'left')]

xs = []
ys = []

for t in subpage['texts']:
    xs.append(t['left'])
    ys.append(t['top'])

plt.hist(xs, max(xs))
plt.hist(ys, max(ys))

sorted(xs)
sorted(ys)

xs_arr = np.array(xs)
ys_arr = np.array(ys)

#%%
def plot_positions_cluster_scatter(ys, clust_ind, clusters_w_vals, clusters_w_inds, cluster_means,
                                   label, label_y_offset):
    cluster_ind_means = {c: np.mean(inds) for c, inds in clusters_w_inds.items()}
    
    plt.figure(figsize=(8, 6))
    ax = plt.axes()
    ax.scatter(range(0, len(ys)), ys, c=clust_ind)
    ax.set_xlabel('index in sorted position list')
    ax.set_ylabel(label + ' position in pixels')
    
    for c, v_mean in cluster_means.items():
        ax.annotate(str(c), xy=(cluster_ind_means[c], v_mean + label_y_offset))
    plt.show()


def plot_hist_with_peaks(v, peak_vals):
    plt.figure(figsize=(8, 6))
    ax = plt.axes()
    ax.hist(v, np.max(v))
    ax.set_xticks(peak_vals)
    plt.show()


#%% Try to find clusters with kmeans
# Sometimes okay, sometimes not
from scipy.cluster.vq import kmeans
import numpy as np

xs_arr = np.array(xs)
xs_sd = np.std(xs_arr)
xs_wh = xs_arr / xs_sd
xs_codebook, xs_dist = kmeans(xs_wh, 5)
xs_peaks = xs_codebook * xs_sd
print(xs_dist)
print(xs_peaks)
plot_hist_with_peaks(xs_arr, xs_peaks)

ys_arr = np.array(ys)
ys_sd = np.std(ys_arr)
ys_wh = ys_arr / ys_sd
ys_codebook, ys_dist = kmeans(ys_wh, 12)
ys_peaks = ys_codebook * ys_sd
print(ys_dist)
print(ys_peaks)
plot_hist_with_peaks(ys_arr, ys_peaks)



#%% Try to find peaks with wavelet transform
# Mostly not correct (parameters issue?)
from scipy.signal import find_peaks_cwt

xs_peaks_ind = find_peaks_cwt(xs_arr, np.arange(5, 10), noise_perc=90)
xs_peaks = xs_arr[xs_peaks_ind]
print(xs_peaks)

plot_hist_with_peaks(xs_arr, xs_peaks)


ys_peaks_ind = find_peaks_cwt(ys_arr, np.arange(30, 50))
ys_peaks = ys_arr[ys_peaks_ind]
print(ys_peaks)

plot_hist_with_peaks(ys_arr, ys_peaks)


#%% Try to find clusters with hierarchical clustering
from scipy.cluster.hierarchy import fclusterdata

from collections import defaultdict

len(ys_arr)
ys_arr.sort()
clust_ind = fclusterdata(ys_arr.reshape((len(ys_arr), 1)), 12,
                         criterion='maxclust',
                         metric='cityblock',
                         method='average')
print(len(np.unique(clust_ind)))
print(clust_ind)

clusters_w_vals = defaultdict(list)
clusters_w_inds = defaultdict(list)
for i, (v, c) in enumerate(zip(ys_arr, clust_ind)):
    clusters_w_vals[c].append(v)
    clusters_w_inds[c].append(i)
cluster_means = {c: np.mean(vals) for c, vals in clusters_w_vals.items()}

clust_ind, clusters_w_vals, clusters_w_inds, cluster_mean_vals = find_best_y_clusters(ys_arr, range(2, 15), mean_dists_range_thresh=30)

plot_positions_cluster_scatter(ys_arr, clust_ind, clusters_w_vals, clusters_w_inds, cluster_mean_vals, 'y', 30)

#%%
def find_best_y_clusters(ys_arr, num_clust_range,
                         mean_dists_range_thresh=float('infinity'),
                         num_vals_per_clust_thresh=float('infinity')):
    """
    Assumptions:
    - y clusters should be equal spaced
    - number of items in y clusters should be equal distributed
    """
    ys_arr.sort()
    RETURN_VALS_ENDIDX = 4
    DIDX = RETURN_VALS_ENDIDX
    NIDX = DIDX + 1
    
    fcluster_runs = []
    for n in num_clust_range:
        clust_ind = fclusterdata(ys_arr.reshape((len(ys_arr), 1)),  # reshape from vector to Nx1 matrix
                                 n,                     # number of clusters to find
                                 criterion='maxclust',  # stop when above n is reached
                                 metric='cityblock',    # 1D distance
                                 method='average')      # average linkage
        assert len(np.unique(clust_ind)) == n
        
        # build dicts with ...        
        clusters_w_vals = defaultdict(list)     # ... cluster -> [values] mapping
        clusters_w_inds = defaultdict(list)     # ... cluster -> [indices] mapping
        for i, (v, c) in enumerate(zip(ys_arr, clust_ind)):
            clusters_w_vals[c].append(v)
            clusters_w_inds[c].append(i)
        
        # calculate mean position value per cluster
        cluster_means = {c: np.mean(vals) for c, vals in clusters_w_vals.items()}
        
        # calculate some properties for minimizing on them later
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
        
        num_vals_per_clust = [len(vals) for vals in clusters_w_vals.values()]
        # vals_per_clust_sd = np.std(num_vals_per_clust)
        vals_per_clust_range = max(num_vals_per_clust) - min(num_vals_per_clust)
        
        if vals_per_clust_range > num_vals_per_clust_thresh:
            continue
        
        print('N=', n,
              # 'dists SD=', mean_dists_sd,
              'dists range=', mean_dists_range,
              # 'num. values SD=', vals_per_clust_sd,
              'num. values range=', vals_per_clust_range)
        
        fcluster_runs.append((clust_ind, clusters_w_vals, clusters_w_inds, cluster_means,
                              mean_dists_range, vals_per_clust_range))
    
    # minimize for mean distance range and number of values per cluster range
    max_dist_range = max(x[DIDX] for x in fcluster_runs)
    #print(max_dist_range)
    max_vals_per_clust_range = max(x[NIDX] for x in fcluster_runs)
    #print(max_vals_per_clust_range)
    best_cluster_runs = sorted(fcluster_runs, key=lambda x: x[DIDX] / max_dist_range + x[NIDX] / max_vals_per_clust_range)
    #best_ns = [len(x[0]) for x in best_cluster_runs]
    #print(best_ns)
    
    return best_cluster_runs[0][0:RETURN_VALS_ENDIDX]

#%%
def find_clusters(arr, n_clust):
    sd = np.std(arr)
    codebook, dist = kmeans(arr / sd, n_clust)     # divide by SD to normalize
    return codebook * sd, dist
    
