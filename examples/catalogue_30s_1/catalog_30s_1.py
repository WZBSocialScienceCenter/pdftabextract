# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:29 2016

@author: mkonrad
"""

import os
from math import radians, degrees
from collections import OrderedDict

import numpy as np
import cv2

from pdftabextract import imgproc
from pdftabextract.clustering import find_clusters_1d_break_dist, calc_cluster_centers_range, zip_clusters_and_values
from pdftabextract.geom import pt
from pdftabextract.common import read_xml, parse_pages, ROTATION, SKEW_X, SKEW_Y
from pdftabextract.fixrotation import rotate_textboxes, deskew_textboxes


DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'ALA1934_RR-excerpt.pdf.xml'

N_COL_BORDERS = 17
MIN_COL_WIDTH = 60


def save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background):
    file_suffix = 'lines-orig' if orig_img_as_background else 'lines'
    
    img_lines = iproc_obj.draw_lines(orig_img_as_background=orig_img_as_background)
    img_lines_file = os.path.join(OUTPUTPATH, '%s-%s.png' % (imgfilebasename, file_suffix))
    
    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)
    

xmltree, xmlroot = read_xml(os.path.join(DATAPATH, INPUT_XML))

pages = parse_pages(xmlroot)
    
vertical_lines_clusters = OrderedDict()

for p_num, p in pages.items():
    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = os.path.join(DATAPATH, p['image'])
    
    print("page %d: detecting lines in image file '%s'" % (p_num, imgfile))
    iproc_obj = imgproc.ImageProc(imgfile)
    
    lines_hough = iproc_obj.detect_lines(canny_low_thresh=50, canny_high_tresh=150, canny_kernel_size=3,
                                         hough_rho_res=1,
                                         hough_theta_res=np.pi/500,
                                         hough_votes_thresh_rel=0.2)
    print("> found %d lines" % len(lines_hough))
    
    save_image_w_lines(iproc_obj, imgfilebasename, True)
    save_image_w_lines(iproc_obj, imgfilebasename, False)
            
    rot_or_skew_type, rot_or_skew_radians = iproc_obj.find_rotation_or_skew(radians(0.5), radians(1))  # uses "lines_hough"
    
    # rotate back or deskew text boxes
    needs_fix = True
    if rot_or_skew_type == ROTATION:
        print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
        rotate_textboxes(p, -rot_or_skew_radians, pt(0, 0))
    elif rot_or_skew_type in (SKEW_X, SKEW_Y):
        print("> deskewing in direction '%s' by %f°" % (rot_or_skew_type, -degrees(rot_or_skew_radians)))
        deskew_textboxes(p, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
    else:
        needs_fix = False
        print("> no page rotation / skew found")
    
    if needs_fix:
        # rotate back or deskew detected lines
        lines_hough = iproc_obj.apply_found_rotation_or_skew(rot_or_skew_type, -rot_or_skew_radians)
        
        save_image_w_lines(iproc_obj, imgfilebasename + '-repaired', True)
        save_image_w_lines(iproc_obj, imgfilebasename + '-repaired', False)
    
    clusters_w_vals = iproc_obj.find_clusters(imgproc.DIRECTION_VERTICAL, find_clusters_1d_break_dist,
                                              dist_thresh=MIN_COL_WIDTH/2)
    print("> found %d clusters" % len(clusters_w_vals))
    
    img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, clusters_w_vals)
    save_img_file = os.path.join(OUTPUTPATH, '%s-vertical-clusters.png' % imgfilebasename)
    print("> saving image with detected vertical clusters to '%s'" % save_img_file)
    cv2.imwrite(save_img_file, img_w_clusters)
    
    vertical_lines_clusters[p_num] = clusters_w_vals

#%%

# 1. filter for pages with clusters whose min/max range is acceptable
# (i.e. the deviation from the median is below a certain threshold)
all_clusters_centers_range = {}
all_clusters_centers = {}
for p_num, clusters_w_vals in vertical_lines_clusters.items():
    all_clusters_centers_range[p_num], all_clusters_centers[p_num] = calc_cluster_centers_range(clusters_w_vals,
                                                                                                return_centers=True)
median_range = np.median(list(all_clusters_centers_range.values()))

good_page_nums = [p_num for p_num, centers_range in all_clusters_centers_range.items()
                  if abs(centers_range - median_range) <= MIN_COL_WIDTH/2]

good_cluster_centers = {p_num: all_clusters_centers[p_num] for p_num in good_page_nums}

# align the cluster centers so that they all start with 0 and create a flat list that contains all centers
centers_norm = []
for p_num, centers in good_cluster_centers.items():
    centers = np.array(centers)
    centers_norm.extend(centers - centers[0])

centers_norm = np.array(centers_norm)
#%%
centers_norm_clusters = zip_clusters_and_values(find_clusters_1d_break_dist(centers_norm, dist_thresh=MIN_COL_WIDTH/2), centers_norm)

center_norm_medians = []
MIN_N_VALUES_PER_CLUSTER_STARTVAL = len(good_page_nums)

for min_n_values in range(MIN_N_VALUES_PER_CLUSTER_STARTVAL, 0, -1):
    for _, vals in centers_norm_clusters:
        if len(vals) >= min_n_values:
            center_norm_medians.append(np.median(vals))
    
        if len(center_norm_medians) == N_COL_BORDERS:
            break
    else:
        continue
    break

    
#%%
#from scipy.cluster.hierarchy import fclusterdata
#from collections import defaultdict
#
#centers_norm_mat = np.array(centers_norm).reshape(len(centers_norm), 1)
#clust_ind = fclusterdata(centers_norm_mat, N_COL_BORDERS, 'maxclust', 'cityblock', method='average')
#clust_dict = defaultdict(list)
#for idx, center in zip(clust_ind, centers_norm):
#    clust_dict[idx].append(center)
#
#center_medians = []
#for centers in clust_dict.values():
#    center_medians.append(np.median(centers))
#
#center_medians = list(sorted(center_medians))

#%%
centers1 = all_clusters_centers[1]
amount_diff = len(centers1) - N_COL_BORDERS

def match_quality_1d(a, b):
    if len(a) != len(b):
        raise ValueError('Both arrays must have the same length')
    
    if len(a) == 0:
        raise ValueError('Arrays must be filled')
    
    if type(a) is not np.ndarray:
        a = np.array(a)
    if type(b) is not np.ndarray:
        b = np.array(b)
    
    return np.sum(np.abs(a - b))

# brute force
import itertools
centers_arr = np.array(centers1)
center_medians_arr = np.array(center_norm_medians)

if amount_diff > 0:    # too many values in centers1
    del_indices_perm = itertools.permutations(range(len(centers1)), amount_diff)
    
    candidates = []
    for del_ind in del_indices_perm:
        candidate_arr = np.delete(centers_arr, del_ind)
        center_medians_w_offset = center_medians_arr + candidate_arr[0]
        quality = match_quality_1d(candidate_arr, center_medians_w_offset)
        candidates.append((candidate_arr, quality))
elif amount_diff < 0:  # too few values in centers1
    candidates = []
    
#%%
for p_num, centers in all_clusters_centers.values():
    pass

#max_n_cluster_centers = max(map(len, good_cluster_centers.values()))

#%%

values = [
    [0, 10,     30,         40],
    [0, 11,     29,         42],
    [0,  9, 15, 29, 32,     41],
    [0, 10,     29, 35, 36, 40],
    [0, 12,     32,         43],
    [0,  9,                 41],
]


#%%
    
    
# save repaired XML (i.e. XML with deskewed textbox positions)
repaired_xmlfile = os.path.join(OUTPUTPATH, INPUT_XML[:INPUT_XML.rindex('.')] + '.repaired.xml')

print("> saving repaired XML file to '%s'" % repaired_xmlfile)
xmltree.write(repaired_xmlfile)
