# -*- coding: utf-8 -*-
"""
Image processing functions.

Created on Wed Dec 14 09:51:20 2016

@author: mkonrad
"""

from logging import warning
from math import degrees, radians

import numpy as np
import cv2

from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y
from pdftabextract.geom import normalize_angle, project_polarcoord_lines


DIRECTION_HORIZONTAL = 'h'
DIRECTION_VERTICAL = 'v'

PIHLF = np.pi / 2
PI4TH = np.pi / 4


class ImageProc:
    DRAW_LINE_WIDTH = 2
    
    def __init__(self, imgfile):
        if not imgfile:
            raise ValueError("paramter 'imgfile' must be a non-empty, non-None string")
        
        self.imgfile = imgfile
        self.input_img = None
        self.img_w = None
        self.img_h = None
                
        self.gray_img = None
        self.edges = None     # edges detected by Canny algorithm
        
        self.lines_hough = []   # contains tuples (rho, theta, theta_norm, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL)
        
        self._load_imgfile()

        
    def detect_lines(self, canny_low_thresh, canny_high_tresh, canny_kernel_size,
                     hough_rho_res, hough_theta_res, hough_votes_thresh_rel,
                     hough_votes_thresh_abs=None,
                     gray_conversion=cv2.COLOR_BGR2GRAY):
        """
        Detect lines in input image using hough transform.
        """
        
        self.gray_img = cv2.cvtColor(self.input_img, gray_conversion)
        self.edges = cv2.Canny(self.gray_img, canny_low_thresh, canny_high_tresh, apertureSize=canny_kernel_size)
        
        votes_thresh = hough_votes_thresh_abs if hough_votes_thresh_abs else round(self.img_w * hough_votes_thresh_rel)
        
        # detect lines with hough transform
        lines = cv2.HoughLines(self.edges, hough_rho_res, hough_theta_res, votes_thresh)
        
        self.lines_hough = self._generate_hough_lines(lines)
        
        return self.lines_hough
    
    def draw_lines(self, orig_img_as_background=True):
        lines_ab = self.ab_lines_from_hough_lines(self.lines_hough)
        
        if orig_img_as_background:
            baseimg = np.copy(self.input_img)
        else:
            baseimg = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        
        for p1, p2, line_dir in lines_ab:
            p1 = tuple(p1)
            p2 = tuple(p2)
            line_color = (0, 255, 0) if line_dir == DIRECTION_HORIZONTAL else (0, 0, 255)
            
            cv2.line(baseimg, p1, p2, line_color, self.DRAW_LINE_WIDTH)
        
        return baseimg
    
    def apply_found_rotation_or_skew(self, rot_or_skew_type, rot_or_skew_radians):        
        if rot_or_skew_type is None or rot_or_skew_radians is None:
            return self.lines_hough
        
        if rot_or_skew_type == ROTATION:
            filter_line_dir = None
        else:  # skew
            filter_line_dir = DIRECTION_HORIZONTAL if rot_or_skew_type == SKEW_Y else DIRECTION_VERTICAL
        
        lines_hough_deskewed = []
        for rho, theta, theta_norm, line_dir in self.lines_hough:
            if filter_line_dir is None or (filter_line_dir is not None and line_dir == filter_line_dir):
                theta += rot_or_skew_radians
                theta_norm = normalize_angle(theta)
            
            lines_hough_deskewed.append((rho, theta, theta_norm, line_dir))
        
        self.lines_hough = lines_hough_deskewed
        
        return self.lines_hough

    
    def ab_lines_from_hough_lines(self, lines_hough):
        projected = project_polarcoord_lines([l[:2] for l in lines_hough], self.img_w, self.img_h)
        return [(p1, p2, line_dir) for (p1, p2), (_, _, _, line_dir) in zip(projected, lines_hough)]
    
    def find_rotation_or_skew(self, rot_thresh, rot_same_dir_thresh):
        """
        Find page rotation or horizontal/vertical skew using detected lines in <lines>. The lines list must consist
        of arrays with the line rotation "theta" at array index 1 like the returned list from detect_lines().
        <rot_thresh> is the minimum threshold in radians for a rotation to be counted as such.
        <rot_same_dir_thresh> is the maximum threshold for the difference between horizontal and vertical line
        rotation.
        """
        if not self.lines_hough:
            raise ValueError("no lines present. did you run detect_lines()?")
        
        # get the deviations
        
        hori_deviations = []   # deviation from unity vector in x-direction
        vert_deviations = []   # deviation from unity vector in y-direction
        
        for _, _, theta_norm, line_dir in self.lines_hough:                        
            if line_dir == DIRECTION_VERTICAL:
                deviation = -theta_norm
                if deviation < -PIHLF:
                    deviation += np.pi
                vert_deviations.append(-deviation)
            else:
                deviation = PIHLF - theta_norm
                hori_deviations.append(-deviation)
                
            assert abs(deviation) <= PI4TH

        # get the medians

        if hori_deviations:
            median_hori_dev = np.median(hori_deviations)
        else:
            warning('no horizontal lines found')
            median_hori_dev = 0
        
        if vert_deviations:
            median_vert_dev = np.median(vert_deviations)
        else:
            median_vert_dev = 0
            warning('no vertical lines found')
        
        hori_rot_above_tresh = abs(median_hori_dev) > rot_thresh
        vert_rot_above_tresh = abs(median_vert_dev) > rot_thresh
        
        if hori_rot_above_tresh and vert_rot_above_tresh:
            if abs(median_hori_dev - median_vert_dev) < rot_same_dir_thresh:
                return ROTATION, (median_hori_dev + median_vert_dev) / 2
            else:
                warning('horizontal / vertical rotation not in same direction (%f / %f)'
                      % (degrees(median_hori_dev), degrees(median_vert_dev)))
        elif hori_rot_above_tresh:
            return SKEW_Y, median_hori_dev
        elif vert_rot_above_tresh:
            return SKEW_X, median_vert_dev

        return None, None
    
    def _load_imgfile(self):        
        self.input_img = cv2.imread(self.imgfile)
        if self.input_img is None:
            raise IOError("could not load file '%s'" % self.imgfile)

        self.img_h, self.img_w = self.input_img.shape[:2]
    
    def _generate_hough_lines(self, lines):        
        lines_hough = []

        for l in lines:
            rho, theta = l[0]  # they come like this from OpenCV's hough transform
            theta_norm = normalize_angle(theta)
                
            if abs(PIHLF - theta_norm) > PI4TH:  # vertical
                line_dir = DIRECTION_VERTICAL
            else:
                line_dir = DIRECTION_HORIZONTAL
                        
            lines_hough.append((rho, theta, theta_norm, line_dir))
        
        return lines_hough
