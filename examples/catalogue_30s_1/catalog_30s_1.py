# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:29 2016

@author: mkonrad
"""

import os
from math import radians

import cv2

from pdftabextract import imgproc
from pdftabextract.geom import pt
from pdftabextract.common import read_xml, parse_pages, ROTATION, SKEW_X, SKEW_Y
from pdftabextract.fixrotation import rotate_back, deskew


DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'ALA1934_RR-excerpt.pdf.xml'

xmltree, xmlroot = read_xml(os.path.join(DATAPATH, INPUT_XML))

pages = parse_pages(xmlroot)

for p_num, p in pages.items():
    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = os.path.join(DATAPATH, p['image'])
    
    print("page %d: detecting lines in image file '%s'" % (p_num, imgfile))
    iproc_obj = imgproc.ImageProc(imgfile)
    
    lines_hough, lines_ab, img_lines, img_on_orig = iproc_obj.detect_lines(return_output_images=True)
    print("> found %d lines" % len(lines_hough))
    
    img_lines_file = os.path.join(OUTPUTPATH, imgfilebasename + '-lines.png')
    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)
    
    img_on_orig_file = os.path.join(OUTPUTPATH, imgfilebasename + '-lines-orig.png')
    print("> saving image with detected lines projected on input image to '%s'" % img_on_orig_file)
    cv2.imwrite(img_on_orig_file, img_on_orig)
    
#    rot_or_skew_type, rot_or_skew_radians = iproc_obj.find_rotation_or_skew(lines, radians(0.5), radians(1))
#    
#    if rot_or_skew_type == ROTATION:
#        print("> rotating back by %f" % -rot_or_skew_radians)
#        rotate_back(p, -rot_or_skew_radians, pt(0, 0))
#    elif rot_or_skew_type in (SKEW_X, SKEW_Y):
#        print("> deskewing in direction '%s' by %f" % (rot_or_skew_type, -rot_or_skew_radians))
#        deskew(p, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
#    else:
#        print("> no page rotation / skew found")
#    
## save repaired XML
#repaired_xmlfile = os.path.join(OUTPUTPATH, INPUT_XML[:INPUT_XML.rindex('.')] + '.repaired.xml')
#
#print("> saving repaired XML file to '%s'" % repaired_xmlfile)
#xmltree.write(repaired_xmlfile)
