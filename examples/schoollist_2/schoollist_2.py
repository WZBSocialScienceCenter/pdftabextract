# -*- coding: utf-8 -*-
"""

Feb. 2017, WZB Berlin Social Science Center - https://wzb.eu

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
from pdftabextract.common import read_xml, parse_pages, save_page_grids, DIRECTION_HORIZONTAL, DIRECTION_VERTICAL
from pdftabextract.textboxes import rotate_textboxes, sorted_by_attr, border_positions_from_texts
from pdftabextract.clustering import (find_clusters_1d_break_dist,
                                      calc_cluster_centers_1d,
                                      zip_clusters_and_values)
from pdftabextract.splitpages import split_page_texts, create_split_pages_dict_structure
from pdftabextract.extract import make_grid_from_positions, fit_texts_into_grid, datatable_to_dataframe


#%% Some constants
DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'schoollist_2.pdf.xml'

N_COLS = 3              # number of columns
HEADER_ROW_HEIGHT = 90  # space between the two header row horizontal lines in pixels, measured in the scanned pages
MIN_ROW_GAP = 80        # minimum space between two rows in pixels, measured in the scanned pages
MIN_COL_WIDTH = 410     # minimum space between two columns in pixels, measured in the scanned pages


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

split_texts_and_images = []   # list of tuples with (double page, split text boxes, split images)

for p_num, p in pages.items():
    # get the image file of the scanned page
    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = os.path.join(DATAPATH, p['image'])
    
    print("page %d: splitting double page '%s'..." % (p_num, imgfile))
    
    # create an image processing object with the scanned page
    iproc_obj = imgproc.ImageProc(imgfile)
    
    # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
    page_scaling_x = iproc_obj.img_w / p['width']
    page_scaling_y = iproc_obj.img_h / p['height']
    image_scaling = (page_scaling_x,   # scaling in X-direction
                     page_scaling_y)   # scaling in Y-direction
    
    # in this case, detecting the split line with image processing in the center is hard because it is
    # hardly visible. since the pages were scanned so that they can be split right in the middle, we
    # define the separator line to be in the middle:
    sep_line_img_x = iproc_obj.img_w / 2
    sep_line_page_x = sep_line_img_x / page_scaling_x
    
    # split the scanned double page at the separator line
    split_images = iproc_obj.split_image(sep_line_img_x)
    
    # split the textboxes at the separator line
    split_texts = split_page_texts(p, sep_line_page_x)
    
    split_texts_and_images.append((p, split_texts, split_images))
    
# generate a new XML and "pages" dict structure from the split pages
split_pages_xmlfile = os.path.join(OUTPUTPATH, INPUT_XML[:INPUT_XML.rindex('.')] + '.split.xml')
print("> saving split pages XML to '%s'" % split_pages_xmlfile)
split_tree, split_root, split_pages = create_split_pages_dict_structure(split_texts_and_images,
                                                                        save_to_output_path=split_pages_xmlfile)

# we don't need the original double pages any more, we'll work with 'split_pages'
del pages

#%% Detect clusters of horizontal lines using the image processing module and rotate back or deskew pages

hori_lines_clusters = {}
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
                                                                            omit_on_rot_thresh=radians(0.5),
                                                                            only_direction=DIRECTION_HORIZONTAL)
    
    # rotate back text boxes
    # since often no vertical lines can be detected and hence it cannot be determined if the page is rotated or skewed,
    # we assume that it's always rotated
    if rot_or_skew_type is not None:
        print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
        rotate_textboxes(p, -rot_or_skew_radians, pt(0, 0))
    
        # rotate back detected lines
        lines_hough = iproc_obj.apply_found_rotation_or_skew(rot_or_skew_type, -rot_or_skew_radians)
        
        save_image_w_lines(iproc_obj, imgfilebasename + '-repaired', True)
        save_image_w_lines(iproc_obj, imgfilebasename + '-repaired', False)
    
    # cluster the detected *horizontal* lines using find_clusters_1d_break_dist as simple clustering function
    # (break on distance HEADER_ROW_HEIGHT/2)
    # this is only to find out the header row later
    hori_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_HORIZONTAL, find_clusters_1d_break_dist,
                                            dist_thresh=HEADER_ROW_HEIGHT/2)
    print("> found %d clusters" % len(hori_clusters))
    
    if len(hori_clusters) > 0:
        # draw the clusters
        img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_HORIZONTAL, hori_clusters)
        save_img_file = os.path.join(OUTPUTPATH, '%s-hori-clusters.png' % imgfilebasename)
        print("> saving image with detected horizontal clusters to '%s'" % save_img_file)
        cv2.imwrite(save_img_file, img_w_clusters)
        
        hori_lines_clusters[p_num] = hori_clusters
    else:
        print("> no horizontal line clusters found")

# save split and repaired XML (i.e. XML with deskewed textbox positions)
output_files_basename = INPUT_XML[:INPUT_XML.rindex('.')]
repaired_xmlfile = os.path.join(OUTPUTPATH, output_files_basename + '.split.repaired.xml')

print("saving split and repaired XML file to '%s'..." % repaired_xmlfile)
split_tree.write(repaired_xmlfile)


#%% Determine the rows and columns of the tables

page_grids = {}

p_num = 2
p = split_pages[p_num]

#print("detecting rows and columns...")
#for p_num, p in split_pages.items():
scaling_x, scaling_y = pages_image_scaling[p_num]

# try to find out the table header using the horizontal lines that were detected before
hori_lines = list(np.array(calc_cluster_centers_1d(hori_lines_clusters[p_num])) / scaling_y)

possible_header_lines = [y for y in hori_lines if y < p['height'] * 0.25]   # all line clusters in the top quarter of the page

if len(possible_header_lines) < 2:
    print("> page %d: no table found" % p_num)  # TODO: no table header -> continue

table_top_y = sorted(possible_header_lines)[-1]

table_texts = [t for t in p['texts'] if t['top'] >= table_top_y]

texts_ys = border_positions_from_texts(table_texts, DIRECTION_VERTICAL)
row_clusters = zip_clusters_and_values(find_clusters_1d_break_dist(texts_ys, dist_thresh=MIN_ROW_GAP/2/scaling_y),
                                       texts_ys)

row_positions = []
prev_row_bottom = None
for _, row_ys in row_clusters:
    row_top = np.min(row_ys)
    row_bottom = np.max(row_ys)
    
    if not row_positions:
        row_positions.append(row_top)
    else:
        row_positions.append(row_top - (row_top - prev_row_bottom)/2)
    
    prev_row_bottom = row_bottom
    
#row_positions.append(prev_row_bottom)

in_rows_texts = [t for t in table_texts if t['bottom'] <= row_positions[-1]]
texts_xs = border_positions_from_texts(in_rows_texts, DIRECTION_HORIZONTAL, only_attr='low')  # left borders of text boxes

col_clusters = zip_clusters_and_values(find_clusters_1d_break_dist(texts_xs, dist_thresh=15),
                                       texts_xs)
col_cluster_sizes = map(lambda x: len(x[0]), col_clusters)
col_clusters_by_size = sorted(zip(col_clusters, col_cluster_sizes), key=lambda x: x[1], reverse=True)

col_positions = []
for (_, col_xs), _ in col_clusters_by_size[:N_COLS]:
    col_positions.append(np.min(col_xs))
col_positions = sorted(col_positions)

last_col_texts = [t for t in in_rows_texts if col_positions[-1] <= t['left'] < p['width']]
col_positions.append(max([t['right'] for t in last_col_texts]))

# create the grid
grid = make_grid_from_positions(col_positions, row_positions)

n_rows = len(grid)
n_cols = len(grid[0])
print("> page %d: grid with %d rows, %d columns" % (p_num, n_rows, n_cols))

page_grids[p_num] = grid

page_grids_file = os.path.join(OUTPUTPATH, output_files_basename + '.pagegrids.json')
print("saving page grids JSON file to '%s'" % page_grids_file)
save_page_grids(page_grids, page_grids_file)
