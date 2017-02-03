# -*- coding: utf-8 -*-
"""
Dec. 2016, WZB Berlin Social Science Center - https://wzb.eu

@author: Markus Konrad <markus.konrad@wzb.eu>
"""

import os
import re
from math import radians, degrees

import numpy as np
import pandas as pd
import cv2

from pdftabextract import imgproc
from pdftabextract.geom import pt
from pdftabextract.common import (read_xml, parse_pages, save_page_grids, all_a_in_b,
                                  ROTATION, SKEW_X, SKEW_Y, DIRECTION_VERTICAL)
from pdftabextract.textboxes import (border_positions_from_texts, split_texts_by_positions, join_texts,
                                     rotate_textboxes, deskew_textboxes)
from pdftabextract.clustering import (find_clusters_1d_break_dist,
                                      calc_cluster_centers_1d,
                                      zip_clusters_and_values,
                                      get_adjusted_cluster_centers)
from pdftabextract.splitpages import split_page_texts, create_split_pages_dict_structure


#%% Some constants
DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'schoollist_1.pdf.xml'

N_COL_BORDERS = 7
MIN_COL_WIDTH = 194

#%% Some helper functions
def save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background, file_suffix_prefix=''):
    file_suffix = 'lines-orig' if orig_img_as_background else 'lines'
    
    img_lines = iproc_obj.draw_lines(orig_img_as_background=orig_img_as_background)
    img_lines_file = os.path.join(OUTPUTPATH, '%s-%s.png' % (imgfilebasename, file_suffix_prefix + file_suffix))
    
    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)

#%% Read the XML

# Load the XML that was generated with pdftohtml
xmltree, xmlroot = read_xml(os.path.join(DATAPATH, INPUT_XML))

# parse it and generate a dict of pages
pages = parse_pages(xmlroot, require_image=True)

#%% Split the scanned double pages so that we can later process the lists page-by-page

split_pages = []

p_num = 1
p = pages[p_num]

# get the image file of the scanned page
imgfilebasename = p['image'][:p['image'].rindex('.')]
imgfile = os.path.join(DATAPATH, p['image'])

print("page %d: detecting lines in image file '%s'..." % (p_num, imgfile))

# create an image processing object with the scanned page
iproc_obj = imgproc.ImageProc(imgfile)

# calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
page_scaling_x = iproc_obj.img_w / p['width']
page_scaling_y = iproc_obj.img_h / p['height']
image_scaling = (page_scaling_x,   # scaling in X-direction
                 page_scaling_y)   # scaling in Y-direction

# detect the lines in the double pages
lines_hough = iproc_obj.detect_lines(canny_low_thresh=50, canny_high_thresh=150, canny_kernel_size=3,
                                     hough_rho_res=1,
                                     hough_theta_res=np.pi/500,
                                     hough_votes_thresh=350)
print("> found %d lines" % len(lines_hough))

save_image_w_lines(iproc_obj, imgfilebasename, True, 'bothpages-')

# find the vertical line that separates both sides
sep_line_img_x = iproc_obj.find_pages_separator_line(dist_thresh=MIN_COL_WIDTH/2)
sep_line_page_x = sep_line_img_x / page_scaling_x
print("> found pages separator line at %f (image space position) / %f (page space position)"
      % (sep_line_img_x, sep_line_page_x))

# split the scanned double page at the separator line
split_images = iproc_obj.split_image(sep_line_img_x)

# split the textboxes at the separator line
split_texts = split_page_texts(p, sep_line_page_x)

split_pages.append((p, split_texts, split_images))

# generate a new XML and "pages" dict structure from the split pages
split_pages_xmlfile = os.path.join(OUTPUTPATH, INPUT_XML[:INPUT_XML.rindex('.')] + '.split.xml')
print("> saving split pages XML to '%s'" % split_pages_xmlfile)
split_tree, split_root, split_pages = create_split_pages_dict_structure(split_pages,
                                                                        save_to_output_path=split_pages_xmlfile)

# we don't need the original double pages any more, we'll work with 'split_pages'
del pages

#%% Detect clusters of vertical lines using the image processing module and rotate back or deskew pages

vertical_lines_clusters = {}
pages_image_scaling = {}     # scaling of the scanned page image in relation to the OCR page dimensions for each page

for p_num, p in split_pages.items():
    # get the image file of the scanned page
    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = os.path.join(OUTPUTPATH, p['image'])
    
    print("page %d: detecting lines in image file '%s'..." % (p_num, imgfile))
    
    # create an image processing object with the scanned page
    iproc_obj = imgproc.ImageProc(imgfile)
    
    # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
    page_scaling_x = iproc_obj.img_w / p['width']
    page_scaling_y = iproc_obj.img_h / p['height']
    pages_image_scaling[p_num] = (page_scaling_x,   # scaling in X-direction
                                  page_scaling_y)   # scaling in Y-direction
    
    # detect the lines
    lines_hough = iproc_obj.detect_lines(canny_low_thresh=50, canny_high_thresh=150, canny_kernel_size=3,
                                         hough_rho_res=1,
                                         hough_theta_res=np.pi/500,
                                         hough_votes_thresh=round(0.2 * iproc_obj.img_w))
    print("> found %d lines" % len(lines_hough))
    
    save_image_w_lines(iproc_obj, imgfilebasename, True)
    save_image_w_lines(iproc_obj, imgfilebasename, False)
    
    # find rotation or skew
    # the parameters are:
    # 1. the minimum threshold in radians for a rotation to be counted as such
    # 2. the maximum threshold for the difference between horizontal and vertical line rotation (to detect skew)
    # 3. an optional threshold to filter out "stray" lines whose angle is too far apart from the median angle of
    #    all other lines that go in the same direction (no effect here)
    rot_or_skew_type, rot_or_skew_radians = iproc_obj.find_rotation_or_skew(radians(0.5),    # uses "lines_hough"
                                                                            radians(1),
                                                                            omit_on_rot_thresh=radians(0.5))
    
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
    
    # cluster the detected *vertical* lines using find_clusters_1d_break_dist as simple clustering function
    # (break on distance MIN_COL_WIDTH/2)
    # additionaly, remove all cluster sections that are considered empty
    # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
    # per cluster section
    vertical_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_VERTICAL, find_clusters_1d_break_dist,
                                                remove_empty_cluster_sections_use_texts=p['texts'], # use this page's textboxes
                                                remove_empty_cluster_sections_n_texts_ratio=0.1,    # 10% rule
                                                remove_empty_cluster_sections_scaling=page_scaling_x,  # the positions are in "scanned image space" -> we scale them to "text box space"
                                                dist_thresh=MIN_COL_WIDTH/2)
    print("> found %d clusters" % len(vertical_clusters))
    
    if len(vertical_clusters) > 0:
        # draw the clusters
        img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
        save_img_file = os.path.join(OUTPUTPATH, '%s-vertical-clusters.png' % imgfilebasename)
        print("> saving image with detected vertical clusters to '%s'" % save_img_file)
        cv2.imwrite(save_img_file, img_w_clusters)
        
        vertical_lines_clusters[p_num] = vertical_clusters
    else:
        print("> no vertical line clusters found")
