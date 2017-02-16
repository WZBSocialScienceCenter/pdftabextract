# -*- coding: utf-8 -*-
"""
An example script that shows how to extract tabular data from OCR-scanned *double* pages with lists of public
schools in Germany.

It includes the following stages:
1. Load the XML describing the pages and text boxes (the XML was generated from the OCR scanned PDF with poppler
   utils (pdftohtml command))
2. Split the scanned double pages so that we can later process the lists page-by-page
3. Detect clusters of horizontal lines using the image processing module and repair rotated pages
4. Get column and line positions of all pages (for lines/rows using the detected horizontal lines and for columns
   by analyzing the distribution of text box x-positions)
5. Create a grid of columns and lines for each page
6. Match the text boxes into the grid and hence extract the tabular data, storing it into a pandas DataFrame

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
from pdftabextract.common import read_xml, parse_pages, save_page_grids
from pdftabextract.textboxes import rotate_textboxes, sorted_by_attr
from pdftabextract.clustering import (find_clusters_1d_break_dist,
                                      calc_cluster_centers_1d,
                                      zip_clusters_and_values)
from pdftabextract.splitpages import split_page_texts, create_split_pages_dict_structure
from pdftabextract.extract import make_grid_from_positions, fit_texts_into_grid, datatable_to_dataframe


#%% Some constants
DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'schoollist_1.pdf.xml'

MIN_ROW_HEIGHT = 260  # <- very important. the minimum height of a row in pixels, measured in the scanned pages
MIN_COL_WIDTH = 194   # <- very important. the minimum width of a column in pixels, measured in the scanned pages

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
                                                                            omit_on_rot_thresh=radians(0.5))
    
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
    # (break on distance MIN_ROW_HEIGHT/2)
    # additionaly, remove all cluster sections that are considered empty
    # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
    # per cluster section
    hori_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_HORIZONTAL, find_clusters_1d_break_dist,
                                            remove_empty_cluster_sections_use_texts=p['texts'], # use this page's textboxes
                                            remove_empty_cluster_sections_n_texts_ratio=0.1,    # 10% rule
                                            remove_empty_cluster_sections_scaling=page_scaling_y,  # the positions are in "scanned image space" -> we scale them to "text box space"
                                            dist_thresh=MIN_ROW_HEIGHT/2)
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

pttrn_schoolnum = re.compile(r'^\d{6}$')   # a valid school number indicates a table row
page_grids = {}

print("detecting rows and columns...")
for p_num, p in split_pages.items():
    scaling_x, scaling_y = pages_image_scaling[p_num]
    
    # try to find out the table rows in this page using the horizontal lines that were detected before
    hori_lines = list(np.array(calc_cluster_centers_1d(hori_lines_clusters[p_num])) / scaling_y)
    hori_lines.append(p['height'])  # last line: page bottom
    
    prev_line_y = 0
    row_texts = []
    row_positions = []
    in_table = False   # is True when the current segment is a real table row (not a table header or surrounding text)
    for line_y in hori_lines:
        # get all texts in this row
        segment_texts = [t for t in p['texts'] if prev_line_y < t['bottom'] <= line_y]
        
        if not segment_texts: continue  # skip empty rows
        
        # try to find the start and the end of the table
        for t in segment_texts:
            t_val = t['value'].strip()
            if pttrn_schoolnum.search(t_val):   # if this matches, we found the start of the table
                if not in_table:
                    in_table = True
                    row_positions.append(prev_line_y)
                break
        else:
            if in_table:   # we found the end of the table
                in_table = False
        
        if in_table:  # this is a table row, so add the texts and row positions to the respective lists
            row_texts.append(segment_texts)
            row_positions.append(line_y)
        
        prev_line_y = line_y
    
    # try to find out the table columns in this page using the distribution of x-coordinates of the left position of
    # each text box in all rows
    text_xs = []
    for texts in row_texts:
        text_xs.extend([t['left'] for t in texts])
    
    text_xs = np.array(text_xs)
    
    # make clusters of x positions
    text_xs_clusters = find_clusters_1d_break_dist(text_xs, dist_thresh=MIN_COL_WIDTH/2/scaling_x)
    text_xs_clusters_w_values = zip_clusters_and_values(text_xs_clusters, text_xs)
    col_positions = calc_cluster_centers_1d(text_xs_clusters_w_values)
    
    # remove falsely identified columns (i.e. merge columns with only a few text boxes)
    filtered_col_positions = []
    n_rows = len(row_positions)
    n_cols = len(col_positions)
    if n_cols > 1 and n_rows > 1:
        top_y = row_positions[0]
        bottom_y = row_positions[-1]
        
        # append the rightmost text's right border as the last column border
        rightmost_pos = sorted_by_attr(p['texts'], 'right')[-1]['right']
        col_positions.append(rightmost_pos)
        
        # merge columns with few text boxes
        texts_in_table = [t for t in p['texts'] if top_y < t['top'] + t['height']/2 <= bottom_y]
        prev_col_x = col_positions[0]
        for col_x in col_positions[1:]:
            col_texts = [t for t in texts_in_table if prev_col_x < t['left'] + t['width']/2 <= col_x]

            if len(col_texts) >= n_rows:   # there should be at least one text box per row
                filtered_col_positions.append(prev_col_x)
                last_col_x = col_x
            prev_col_x = col_x
        
        # manually add border for the last column because it has very few or no text boxes
        filtered_col_positions.append(filtered_col_positions[-1] + (rightmost_pos - filtered_col_positions[-1]) / 2)
        filtered_col_positions.append(rightmost_pos)

    # create the grid
    if filtered_col_positions:
        grid = make_grid_from_positions(filtered_col_positions, row_positions)
        
        n_rows = len(grid)
        n_cols = len(grid[0])
        print("> page %d: grid with %d rows, %d columns" % (p_num, n_rows, n_cols))
        
        page_grids[p_num] = grid
    else:  # this happens for the first page as there's no table on that
        print("> page %d: no table found" % p_num)
    
# save the page grids

# After you created the page grids, you should then check that they're correct using pdf2xml-viewer's 
# loadGridFile() function

page_grids_file = os.path.join(OUTPUTPATH, output_files_basename + '.pagegrids.json')
print("saving page grids JSON file to '%s'" % page_grids_file)
save_page_grids(page_grids, page_grids_file)

#%% Create data frames (requires pandas library)

# For sake of simplicity, we will just fit the text boxes into the grid, merge the texts in their cells (splitting text
# boxes to separate lines if necessary) and output the result. Normally, you would do some more parsing here, e.g.
# extracting the adress components from the second column.

full_df = pd.DataFrame()
print("fitting text boxes into page grids and generating final output...")
for p_num, p in split_pages.items():
    if p_num not in page_grids: continue  # happens when no table was detected

    print("> page %d" % p_num)
    datatable, unmatched_texts = fit_texts_into_grid(p['texts'], page_grids[p_num], return_unmatched_texts=True)
    
    df = datatable_to_dataframe(datatable, split_texts_in_lines=True)
    df['from_page'] = p_num
    full_df = full_df.append(df, ignore_index=True)

print("extracted %d rows from %d pages" % (len(full_df), len(split_pages)))

csv_output_file = os.path.join(OUTPUTPATH, output_files_basename + '.csv')
print("saving extracted data to '%s'" % csv_output_file)
full_df.to_csv(csv_output_file, index=False)

excel_output_file = os.path.join(OUTPUTPATH, output_files_basename + '.xlsx')
print("saving extracted data to '%s'" % excel_output_file)
full_df.to_excel(excel_output_file, index=False)
