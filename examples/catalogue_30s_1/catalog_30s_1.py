# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:29 2016

@author: mkonrad
"""

import os
from math import radians, degrees

import numpy as np
import cv2

from pdftabextract import imgproc
from pdftabextract.geom import pt
from pdftabextract.fixrotation import rotate_textboxes, deskew_textboxes
from pdftabextract.textboxes import border_positions_and_spans_from_texts
from pdftabextract.clustering import (find_clusters_1d_break_dist,
                                      calc_cluster_centers_1d,
                                      zip_clusters_and_values,
                                      get_adjusted_cluster_centers)
from pdftabextract.common import (read_xml, parse_pages,
                                  ROTATION, SKEW_X, SKEW_Y,
                                  DIRECTION_HORIZONTAL, DIRECTION_VERTICAL)


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
    
vertical_lines_clusters = {}
pages_image_scaling = {}     # scaling of the scanned page image in relation to the OCR page dimensions for each page

for p_num, p in pages.items():
    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = os.path.join(DATAPATH, p['image'])
    
    print("page %d: detecting lines in image file '%s'" % (p_num, imgfile))
    iproc_obj = imgproc.ImageProc(imgfile)
    
    pages_image_scaling[p_num] = (iproc_obj.img_w / p['width'], iproc_obj.img_h / p['height'])
    
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
# save repaired XML (i.e. XML with deskewed textbox positions)
repaired_xmlfile = os.path.join(OUTPUTPATH, INPUT_XML[:INPUT_XML.rindex('.')] + '.repaired.xml')

print("> saving repaired XML file to '%s'" % repaired_xmlfile)
xmltree.write(repaired_xmlfile)

    
#%% Get column positions as adjusted vertical line clusters
print("> calculating column positions for all pages")

pages_image_scaling_x = {p_num: sx for p_num, (sx, _) in pages_image_scaling.items()}

col_positions = get_adjusted_cluster_centers(vertical_lines_clusters, N_COL_BORDERS,
                                             max_range_deviation=MIN_COL_WIDTH/2,
                                             find_center_clusters_method=find_clusters_1d_break_dist,
                                             dist_thresh=MIN_COL_WIDTH/2,
                                             image_scaling=pages_image_scaling_x)   # the positions are in "scanned
                                                                                    # image space" -> we scale them
                                                                                    # to "text box space"

#%% Get line positions
print("> calculating line positions for all pages")
line_positions = {}

for p_num, p in pages.items():
    borders_y, text_heights = border_positions_and_spans_from_texts(p['texts'], DIRECTION_VERTICAL)
    median_text_height = np.median(text_heights)
    
    hori_clusters = find_clusters_1d_break_dist(borders_y, dist_thresh=median_text_height/2)
    hori_clusters_w_vals = zip_clusters_and_values(hori_clusters, borders_y)
    
    line_positions = calc_cluster_centers_1d(hori_clusters_w_vals)
    line_heights = np.diff(line_positions)
    
    print("> page %d: %d lines, median text height = %f, median line height = %f, min line height = %f, max line height = %f"
          % (p_num, len(line_positions), median_text_height, np.median(line_heights), min(line_heights), max(line_heights)))

