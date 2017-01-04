# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:11:33 2016

@author: mkonrad
"""


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

#override_angles = {
#    (32, 'right'): -1
#}
override_angles = None

xmltree, xmlroot, rot_results = fixrotation.fix_rotation('testxmls/1992_93_neu.pdf.xml',
                                                         corner_box_cond_fns,
                                                         override_angles=override_angles)
#%%
xmltree.write('testxmls/1992_93_neu_rotfixed.pdf.xml')

#%%

from pdftabextract.common import read_xml, parse_pages, get_bodytexts, divide_texts_horizontally

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5
# DIVIDE_RATIO = None

LEFTMOST_COL_ALIGN = 'topleft'     # topleft, topright or center
RIGHTMOST_COL_ALIGN = 'topleft'    # topleft, topright or center

MIN_CONTENTLENGTH_MEAN_DEV_RATIO = 0.2

tree, root = read_xml('testxmls/1992_93_neu.pdf.xml')

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

def plot_positions_scatter2(pos, clust_ind=None, cluster_stats=None, ylim=None):   
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    cols = clust_ind if clust_ind is not None else None
    ax.scatter(range(0, len(pos)), pos, c=cols)
    ax.set_xlabel('index in sorted position list')
    if ylim:
        ax.set_ylim(ylim)
    
    if cluster_stats:
        for i, (n, x, y) in enumerate(cluster_stats):
            ax.annotate(str(i) + ' n=' + str(n), xy=(x, y - 30))
    
    plt.show()


#%%

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from pdftabextract.clustering import zip_clusters_and_values

from collections import defaultdict


def create_cluster_dicts(vals, clust_ind):
    # build dicts with ...        
    clusters_w_vals = defaultdict(list)     # ... cluster -> [values] mapping
    clusters_w_inds = defaultdict(list)     # ... cluster -> [indices] mapping
    for i, (v, c) in enumerate(zip(vals, clust_ind)):
        clusters_w_vals[c].append(v)
        clusters_w_inds[c].append(i)
    
    return clusters_w_vals, clusters_w_inds


vals = np.array(sorted([124.62822580645161, 175.88169476038982, 434.49032258064517, 484.13940157981807, 509.67699175406989, 535.38714065032207, 561.11187502302857, 587.71290322580649, 612.54675829198709, 638.08437041920649, 663.62196059345831, 688.42258064516125, 714.68809402836439, 742.01451612903224, 773.66612903225803, 799.56290322580639, 824.40093367401096, 124.62822580645161, 175.88302388911299, 334.68374616024266, 433.95665911735045, 486.28744484529938, 511.6485164485033, 537.53815696473453, 562.72367612668484, 588.61235051312156, 614.86880212602478, 640.05589511363985, 665.94299567441192, 690.40106019054087, 716.30342227704864, 741.83493115828276, 774.20589890021824, 800.46235051312146, 825.65311838643674, 124.62822580645161, 176.4171423804712, 434.66775176892719, 484.6667829797031, 510.55968725279814, 536.8161388657013, 562.71291305924979, 588.2500098334433, 613.25482578253809, 640.04355822054004, 664.50936773934359, 688.61597195005481, 714.68082012902676, 743.09560812710231, 774.21281569287964, 799.75040586713146, 824.93236739284077, 124.62822580645161, 176.24171262560836, 433.0564589793978, 483.94596774193548, 509.12655588923246, 535.91935483870964, 559.658064516129, 585.55483870967737, 611.10029412308836, 637.34838709677422, 664.32419354838714, 687.88306451612902, 714.33782431341501, 741.11532258064517, 771.51958623615337, 786.25483870967741, 797.41687933086018, 823.13208432069621, 124.62822580645161, 175.5225806451613, 433.23687109486821, 484.49733158798176, 512.54032258064512, 537.7177419354839, 563.61451612903227, 588.08648201384142, 615.76774193548385, 639.86612903225807, 665.41092756736282, 689.50955426790097, 715.77447679103557, 743.99274193548388, 774.74516129032259, 801.36129032258066, 825.1104839177076, 124.62822580645161, 177.68064516129033, 435.21214547327037, 486.29087376417681, 513.61935483870968, 539.1564516129032, 564.69354838709671, 589.88701989020592, 615.77601470964828, 640.94516129032252, 665.23214262659906, 689.68145161290317, 715.2278065193974, 743.112604841652, 774.02580645161288, 786.61451612903227, 798.31411908013115, 823.32225089818462, 124.62822580645161, 175.16389842432878, 432.15849075010385, 484.32081863895945, 509.49872784572375, 534.84995749545749, 560.5658694152404, 586.81982392499549, 612.72011461917396, 637.5423842009119, 664.50714890616882, 688.97322612740948, 715.40577309614605, 743.46087657860176, 775.64747148681397, 801.1845682610076, 825.28831462346102, 124.62822580645161, 174.80322580645162, 204.65645161290323, 432.15241935483868, 484.48548387096776, 510.02258064516127, 534.48732868031243, 560.37741935483871, 586.63387096774193, 612.35894417828877, 637.70806451612907, 663.25380161947419, 687.88306451612902, 713.06961405204504, 740.23492248810157, 771.70731448479899, 797.06508612827452, 822.24301728800629, 124.62822580645161, 177.67956822083812, 433.24241320585759, 485.03697744230362, 510.21488664906803, 535.38473512387793, 561.63629032258063, 587.18335167926875, 612.71691148754326, 638.60725806451615, 663.96451612903229, 687.35548900806259, 714.85887096774195, 742.92271694308465, 773.85556553558331, 798.67381572782779, 826.17903225806447, 890.76112616120588, 124.62822580645161, 174.98306451612902, 432.69193548387096, 483.40931771301905, 509.30635135714664, 535.56710908528908, 561.10040944298441, 585.55483870967737, 611.27177419354837, 636.81729508807757, 661.80645161290317, 686.80403225806447, 713.42933648722931, 740.58524642129771, 771.14838709677417, 796.51564372576422, 822.94193548387091, 124.62822580645161, 177.50281185796374, 434.85152416347626, 487.18509660427992, 512.00941279878509, 536.46392970459738, 562.3612008463366, 587.89577745451993, 613.43640314780782, 638.25463138708471, 665.05113787699395, 691.47983870967744, 717.74491550864241, 744.90110296603905, 776.18875457402601, 801.7312446532444, 826.72934630571638]))
data = vals.reshape((len(vals), 1))

t = 10
ind = fclusterdata(data, t, criterion='distance', method='median')

clusters = [np.where(ind == c_id)[0] for c_id in np.unique(ind)]
clusters_w_vals = zip_clusters_and_values(clusters, vals)
print('len(clusters_w_vals)', len(clusters_w_vals))
cluster_stats = []
for c_ind, c_vals in clusters_w_vals:
    cluster_stats.append((len(c_vals), np.mean(c_ind), np.median(c_vals)))

plot_positions_scatter2(vals, ind, cluster_stats)
    
#%%
import matplotlib.pyplot as plt
import numpy as np

subpage = subpages[(5, 'right')]

xs = []
ys = []

for t in subpage['texts']:
    xs.append(t['left'])
    ys.append(t['top'])

#plt.hist(xs, max(xs))
#plt.hist(ys, max(ys))


plot_positions_scatter(xs, label='x', ylim=(subpage['x_offset'], subpage['x_offset'] + subpage['width']))
plot_positions_scatter(ys, label='y', ylim=(0, subpage['height']))

xs_arr = np.array(xs)
ys_arr = np.array(ys)


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

#%%

ys_arr, clust_ind = find_best_pos_clusters(ys_arr, range(2, 15), 'y', mean_dists_range_thresh=30)
clusters_w_vals, clusters_w_inds = create_cluster_dicts(ys_arr, clust_ind)
cluster_means = calc_cluster_means(clusters_w_vals)

print('found clusters with N=', len(clusters_w_vals))
plot_positions_scatter(ys_arr, clust_ind, clusters_w_vals, clusters_w_inds, cluster_means, 'y', 30)

#%%
xs_arr, clust_ind = find_best_pos_clusters(xs_arr, range(3, 9), 'x', property_weights=(1, 5))

clusters_w_vals, clusters_w_inds = create_cluster_dicts(xs_arr, clust_ind)
cluster_means = calc_cluster_means(clusters_w_vals)

print('found clusters with N=', len(clusters_w_vals))
plot_positions_scatter(xs_arr, clust_ind, clusters_w_vals, clusters_w_inds, cluster_means, 'x', 30)

filtered_clust = {c: vals for c, vals in clusters_w_vals.items() if len(vals) > 3}

#%%
def find_clusters(arr, n_clust):
    sd = np.std(arr)
    codebook, dist = kmeans(arr / sd, n_clust)     # divide by SD to normalize
    return codebook * sd, dist
    

#%%
from random import shuffle

from geom import pt, rect, rectintersect

from logging import warning

col_positions, row_positions = find_col_and_row_positions_in_subpage(subpage)

n_rows = len(row_positions)
n_cols = len(col_positions)

row_ranges = position_ranges(row_positions, subpage['height'])
col_ranges = position_ranges(col_positions, subpage['x_offset'] + subpage['width'])

#%%
grid = {(r_i, c_i): rect(pt(l, t), pt(r, b)) for r_i, (t, b) in enumerate(row_ranges)
                                             for c_i, (l, r) in enumerate(col_ranges)}

table = np.empty((n_rows, n_cols), dtype='object')
# np.full does not with with list as fill value so we have to do it like this:
for j in range(n_rows):
    for k in range(n_cols):
        table[j, k] = []
        
textblocks = subpage['texts'][:]
for t in textblocks[:]:
    t_rect = rect(t['topleft'], t['bottomright'])
    cell_isects = []
    for idx, cell_rect in grid.items():
        isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
        if isect is not None and isect > 0:
            cell_isects.append((idx, isect))
    
    if len(cell_isects) > 0:
        max_isect_val = max([x[1] for x in cell_isects])
        if max_isect_val < 0.5:
            warning("low best cell intersection value: %f" % max_isect_val)
        best_isects = [x for x in cell_isects if x[1] == max_isect_val]
        if len(best_isects) > 1:
            warning("multiple (%d) best cell intersections" % len(best_isects))
        best_idx = best_isects[0][0]
        table[best_idx].append(t)
        textblocks.remove(t)

for t in textblocks:
    warning("no cell found for textblock '%s'" % t)

#%%

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



#%%

cell = table[2, 1]

lines = put_texts_in_lines(cell)
text = create_text_from_lines(lines)


#y_sorted_ts, vspacings = zip(*calc_text_spacings(cell, direction='vertical'))
#x_sorted_ts, hspacings = zip(*calc_text_spacings(cell, direction='horizontal'))

#%%

textmat = np.empty((n_rows, n_cols), dtype='object')

for j in range(n_rows):
    for k in range(n_cols):
        texts = [t['value'] for t in table[j, k]]
        textmat[j, k] = ', '.join(texts)

#%%

layouts = analyze_subpage_layouts(subpages)

#%%

all_row_pos, all_col_pos = zip(*[(np.array(layout[0]), np.array(layout[1]) - subpages[p_id]['x_offset']) for p_id, layout in layouts.items()
                                 if layout is not None])

nrows = [len(row_pos) for row_pos in all_row_pos]
ncols = [len(col_pos) for col_pos in all_col_pos]

nrows_median = np.median(nrows)
ncols_median = np.median(ncols)

invalid_subpages = [p_id for p_id, layout in layouts.items() if layout is None]
sorted(invalid_subpages, key=lambda x: x[0])

col_pos_w_median_len = [col_pos for col_pos in all_col_pos if len(col_pos) == ncols_median]
best_col_pos = [list() for _ in range(int(ncols_median))]
for col_positions in col_pos_w_median_len:
    for i, pos in enumerate(col_positions):
        best_col_pos[i].append(pos)

best_col_pos_medians = [np.median(best_col_pos[i]) for i in range(int(ncols_median))]