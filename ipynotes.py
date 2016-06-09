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
import numpy as np
from scipy.signal import find_peaks_cwt

xs_arr = np.array(xs)
ys_arr = np.array(ys)

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

xs_arr = np.array(xs)
ys_arr = np.array(ys)

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
cluster_ind_means = {c: np.mean(inds) for c, inds in clusters_w_inds.items()}

plt.figure(figsize=(8, 6))
ax = plt.axes()
ax.scatter(range(0, len(ys)), ys_arr, c=clust_ind)
ax.set_xlabel('index in sorted position list')
ax.set_ylabel('y position in pixels')

for c, v_mean in cluster_means.items():
    ax.annotate(str(c), xy=(cluster_ind_means[c], v_mean + 30))
plt.show()

#plt.legend()


#%%
def best_y_clusters_num_clusters_range(ys, num_clust_range, max_dist_thresh=0.1):
    """
    Assumptions:
    - y clusters should be equal spaced
    - number of items in y clusters should be equal distributed
    """
    clusters = []
    for n in num_clust_range:
        clust, dist = find_clusters(ys, n)
        
        if dist > max_dist_thresh:
            continue
        
        # [abs(c_a - c_b) for a, c_a in enumerate(clust) for b, c_b in enumerate(clust) if a != b]
        sorted_clust = list(sorted(clust))
        clust_dist = [c - sorted_clust[i-1] for i, c in enumerate(sorted_clust) if i > 0]
        
        clusters.append(clust)        
        print(n, dist)
    
    



def find_clusters(arr, n_clust):
    sd = np.std(arr)
    codebook, dist = kmeans(arr / sd, n_clust)     # divide by SD to normalize
    return codebook * sd, dist


def plot_hist_with_peaks(v, peak_vals):
    plt.figure(figsize=(8, 6))
    ax = plt.axes()
    ax.hist(v, np.max(v))
    ax.set_xticks(peak_vals)
    plt.show()
    
