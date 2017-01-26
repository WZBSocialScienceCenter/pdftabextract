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

from pdftabextract.common import (read_xml, parse_pages, save_page_grids, all_a_in_b,
                                  ROTATION, SKEW_X, SKEW_Y, DIRECTION_VERTICAL)
from pdftabextract import imgproc
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

# detect the lines
lines_hough = iproc_obj.detect_lines(canny_low_thresh=50, canny_high_thresh=150, canny_kernel_size=3,
                                     hough_rho_res=1,
                                     hough_theta_res=np.pi/500,
                                     hough_votes_thresh=350)
print("> found %d lines" % len(lines_hough))

save_image_w_lines(iproc_obj, imgfilebasename, True, 'bothpages-')

sep_line_img_x = iproc_obj.find_pages_separator_line(dist_thresh=MIN_COL_WIDTH/2)
sep_line_page_x = sep_line_img_x / page_scaling_x
print("> found pages separator line at %f (image space position) / %f (page space position)"
      % (sep_line_img_x, sep_line_page_x))


split_images = iproc_obj.split_image(sep_line_img_x)
split_texts = split_page_texts(p, sep_line_page_x)

split_pages = [
    (p, split_texts, split_images)
]

split_pages_xmlfile = os.path.join(OUTPUTPATH, INPUT_XML[:INPUT_XML.rindex('.')] + '.split.xml')
print("> saving split pages XML to '%s'" % split_pages_xmlfile)
split_tree, split_root, split_pages = create_split_pages_dict_structure(split_pages,
                                                                        save_to_output_path=split_pages_xmlfile)

