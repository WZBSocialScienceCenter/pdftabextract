# -*- coding: utf-8 -*-
"""
An example script that shows how to extract tabular data from OCR-scanned pages of an old catalogue of
German newspapers.

It includes the following stages:
1. Load the XML describing the pages and text boxes (the XML was generated from the OCR scanned PDF with poppler
   utils (pdftohtml command))
2. Detect clusters of vertical lines using the image processing module imgproc
3. Find page rotation or skew and fix it
4. Get column and line positions of all pages
5. Create a grid of columns and lines for each page
6. Match the text boxes into the grid and hence extract the tabular data, storing it into a pandas DataFrame

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
from pdftabextract.textboxes import (border_positions_from_texts, split_texts_by_positions, join_texts,
                                     rotate_textboxes, deskew_textboxes)
from pdftabextract.clustering import (find_clusters_1d_break_dist,
                                      calc_cluster_centers_1d,
                                      zip_clusters_and_values,
                                      get_adjusted_cluster_centers)
from pdftabextract.extract import make_grid_from_positions, fit_texts_into_grid, datatable_to_dataframe
from pdftabextract.common import (read_xml, parse_pages, save_page_grids, all_a_in_b,
                                  ROTATION, SKEW_X, SKEW_Y,
                                  DIRECTION_VERTICAL)

#%% Some constants
DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'ALA1934_RR-excerpt.pdf.xml'

N_COL_BORDERS = 17
MIN_COL_WIDTH = 60   # <- very important! minimum width of a column in pixels, measured in the scanned pages

#%% Some helper functions
def save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background):
    file_suffix = 'lines-orig' if orig_img_as_background else 'lines'
    
    img_lines = iproc_obj.draw_lines(orig_img_as_background=orig_img_as_background)
    img_lines_file = os.path.join(OUTPUTPATH, '%s-%s.png' % (imgfilebasename, file_suffix))
    
    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)

#%% Read the XML

# Load the XML that was generated with pdftohtml
xmltree, xmlroot = read_xml(os.path.join(DATAPATH, INPUT_XML))

# parse it and generate a dict of pages
pages = parse_pages(xmlroot)

#%% Detect clusters of vertical lines using the image processing module and rotate back or deskew pages

vertical_lines_clusters = {}
pages_image_scaling = {}     # scaling of the scanned page image in relation to the OCR page dimensions for each page

for p_num, p in pages.items():
    # get the image file of the scanned page
    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = os.path.join(DATAPATH, p['image'])
    
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
    
    # draw the clusters
    img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
    save_img_file = os.path.join(OUTPUTPATH, '%s-vertical-clusters.png' % imgfilebasename)
    print("> saving image with detected vertical clusters to '%s'" % save_img_file)
    cv2.imwrite(save_img_file, img_w_clusters)
    
    vertical_lines_clusters[p_num] = vertical_clusters

#%%
# save repaired XML (i.e. XML with deskewed textbox positions)
output_files_basename = INPUT_XML[:INPUT_XML.rindex('.')]
repaired_xmlfile = os.path.join(OUTPUTPATH, output_files_basename + '.repaired.xml')

print("saving repaired XML file to '%s'..." % repaired_xmlfile)
xmltree.write(repaired_xmlfile)

    
#%% Get column positions as adjusted vertical line clusters
print("calculating column positions for all pages...")

pages_image_scaling_x = {p_num: sx for p_num, (sx, _) in pages_image_scaling.items()}

col_positions = get_adjusted_cluster_centers(vertical_lines_clusters, N_COL_BORDERS,
                                             find_center_clusters_method=find_clusters_1d_break_dist,
                                             dist_thresh=MIN_COL_WIDTH/2,
                                             image_scaling=pages_image_scaling_x)   # the positions are in "scanned
                                                                                    # image space" -> we scale them
                                                                                    # to "text box space"

#%% Get line positions
print("calculating line positions for all pages...")
line_positions = {}
pttrn_table_row_beginning = re.compile(r'^[\d Oo][\d Oo]{2,} +[A-ZÄÖÜ]')   # a (possibly malformed) population number + space + start of city name
words_in_footer = ('anzeige', 'annahme', 'ala')

for p_num, p in pages.items():
    page_colpos = col_positions[p_num]  # column positions of this page
    col2_rightborder = page_colpos[2]   # right border of the second column
    
    # calculate median text box height
    median_text_height = np.median([t['height'] for t in p['texts']])
    
    # get all texts in the first two columns with a "usual" textbox height
    # we will only use these text boxes in order to determine the line positions because they are more "stable"
    # otherwise, especially the right side of the column header can lead to problems detecting the first table row
    text_height_deviation_thresh = median_text_height / 2
    texts_cols_1_2 = [t for t in p['texts']
                      if t['right'] <= col2_rightborder
                         and abs(t['height'] - median_text_height) <= text_height_deviation_thresh]
    
    # get all textboxes' top and bottom border positions
    borders_y = border_positions_from_texts(texts_cols_1_2, DIRECTION_VERTICAL)
    
    # break into clusters using half of the median text height as break distance
    clusters_y = find_clusters_1d_break_dist(borders_y, dist_thresh=median_text_height/2)
    clusters_w_vals = zip_clusters_and_values(clusters_y, borders_y)
    
    # for each cluster, calculate the median as center
    pos_y = calc_cluster_centers_1d(clusters_w_vals)
    pos_y.append(p['height'])
    
    ### make some additional filtering of the row positions ###
    # 1. try to find the top row of the table
    texts_cols_1_2_per_line = split_texts_by_positions(texts_cols_1_2, pos_y, DIRECTION_VERTICAL,
                                                       alignment='middle',
                                                       enrich_with_positions=True)
    
    # go through the texts line per line
    for line_texts, (line_top, line_bottom) in texts_cols_1_2_per_line:
        line_str = join_texts(line_texts)
        if pttrn_table_row_beginning.match(line_str):  # check if the line content matches the given pattern
            top_y = line_top
            break
    else:
        top_y = 0
    
    # 2. try to find the bottom row of the table
    min_footer_text_height = median_text_height * 1.5
    min_footer_y_pos = p['height'] * 0.7
    # get all texts in the lower 30% of the page that have are at least 50% bigger than the median textbox height
    bottom_texts = [t for t in p['texts']
                    if t['top'] >= min_footer_y_pos and t['height'] >= min_footer_text_height]
    bottom_texts_per_line = split_texts_by_positions(bottom_texts,
                                                     pos_y + [p['height']],   # always down to the end of the page
                                                     DIRECTION_VERTICAL,
                                                     alignment='middle',
                                                     enrich_with_positions=True)
    # go through the texts at the bottom line per line
    page_span = page_colpos[-1] - page_colpos[0]
    min_footer_text_width = page_span * 0.8
    for line_texts, (line_top, line_bottom) in bottom_texts_per_line:
        line_str = join_texts(line_texts)
        has_wide_footer_text = any(t['width'] >= min_footer_text_width for t in line_texts)
        # check if there's at least one wide text or if all of the required words for a footer match
        if has_wide_footer_text or all_a_in_b(words_in_footer, line_str):
            bottom_y = line_top
            break
    else:
        bottom_y = p['height']
    
    # finally filter the line positions so that only the lines between the table top and bottom are left
    pos_y = [y for y in pos_y if top_y <= y <= bottom_y]
    
    # save it to the dict
    line_positions[p_num] = pos_y
    
    # this is just for information:
    line_heights = np.diff(pos_y)
    
    print("> page %d: %d lines between [%f, %f], median text height = %f, median line height = %f, min line height = %f, max line height = %f"
          % (p_num, len(pos_y), top_y, bottom_y,
             median_text_height, np.median(line_heights),
             min(line_heights), max(line_heights)))

#%% Create page grids

# After you created the page grids, you should then check that they're correct using pdf2xml-viewer's 
# loadGridFile() function

print("creating page grids for all pages...")
page_grids = {}
for p_num, p in pages.items():
    grid = make_grid_from_positions(col_positions[p_num], line_positions[p_num])
    n_rows = len(grid)
    n_cols = len(grid[0])
    print("> page %d: grid with %d rows, %d columns" % (p_num, n_rows, n_cols))
    page_grids[p_num] = grid

page_grids_file = os.path.join(OUTPUTPATH, output_files_basename + '.pagegrids.json')
print("saving page grids JSON file to '%s'" % page_grids_file)
save_page_grids(page_grids, page_grids_file)

#%% Create data frames (requires pandas library)

# For sake of simplicity, we will just fit the text boxes into the grid, merge the texts in their cells and
# output the result.
# Normally, at this step you will need to do some error correction / parsing as some text boxes are not correctly
# detected during OCR (they might span over multiple columns or numbers are incorrectly detected as letters).

full_df = pd.DataFrame()
print("fitting text boxes into page grids and generating final output...")
for p_num, p in pages.items():
    print("> page %d" % p_num)
    datatable, unmatched_texts = fit_texts_into_grid(p['texts'], page_grids[p_num], return_unmatched_texts=True)
    
    df = datatable_to_dataframe(datatable)
    df['from_page'] = p_num
    full_df = full_df.append(df, ignore_index=True)

print("extracted %d rows from %d pages" % (len(full_df), len(pages)))

csv_output_file = os.path.join(OUTPUTPATH, output_files_basename + '.csv')
print("saving extracted data to '%s'" % csv_output_file)
full_df.to_csv(csv_output_file, index=False)

excel_output_file = os.path.join(OUTPUTPATH, output_files_basename + '.xlsx')
print("saving extracted data to '%s'" % excel_output_file)
full_df.to_excel(excel_output_file, index=False)
