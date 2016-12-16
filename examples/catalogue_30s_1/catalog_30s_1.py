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
from pdftabextract.common import read_xml, parse_pages, ROTATION, SKEW_X, SKEW_Y
from pdftabextract.fixrotation import rotate_textboxes, deskew_textboxes


DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'ALA1934_RR-excerpt.pdf.xml'

xmltree, xmlroot = read_xml(os.path.join(DATAPATH, INPUT_XML))

pages = parse_pages(xmlroot)

def save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background):
    file_suffix = 'lines-orig' if orig_img_as_background else 'lines'
    
    img_lines = iproc_obj.draw_lines(orig_img_as_background=orig_img_as_background)
    img_lines_file = os.path.join(OUTPUTPATH, '%s-%s.png' % (imgfilebasename, file_suffix))
    
    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)
    

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
    
    
# save repaired XML
repaired_xmlfile = os.path.join(OUTPUTPATH, INPUT_XML[:INPUT_XML.rindex('.')] + '.repaired.xml')

print("> saving repaired XML file to '%s'" % repaired_xmlfile)
xmltree.write(repaired_xmlfile)
