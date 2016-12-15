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
from pdftabextract.geom import pt


DIRECTION_HORIZONTAL = 'h'
DIRECTION_VERTICAL = 'v'

PIHLF = np.pi / 2
PI4TH = np.pi / 4


class ImageProc:    
    def __init__(self, imgfile):
        if not imgfile:
            raise ValueError("paramter 'imgfile' must be a non-empty, non-None string")
        
        self._imgfile = imgfile
        self.input_img = None
        self.img_w = None
        self.img_h = None
                
        self._reset()
    
    def _reset(self):
        self.gray_img = None
        self.edges = None     # edges detected by Canny algorithm
        
        self.lines_hough = []   # contains tuples (rho, theta, theta_norm, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL)
        self.lines_ab = []      # contains tuples (point_a, point_b, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL)        
   
    def detect_lines(self, gray_conversion=cv2.COLOR_BGR2GRAY,
                     canny_low_thresh=50, canny_high_tresh=150, canny_kernel_size=3,
                     hough_rho_res=1,
                     hough_theta_res=np.pi/500,
                     hough_votes_thresh_rel=0.2,
                     hough_votes_thresh_abs=None,
                     return_output_images=False):
        """
        Detect lines in input image using hough transform.
        """
        
        self._load_imgfile()
        self.gray_img = cv2.cvtColor(self.input_img, gray_conversion)
        self.edges = cv2.Canny(self.gray_img, canny_low_thresh, canny_high_tresh, apertureSize=canny_kernel_size)
        
        if return_output_images:
            lines_only_img = np.zeros((self.img_h, self.img_w, 1), np.uint8)
            input_img_copy = np.copy(self.input_img)
        else:
            lines_only_img = None
            input_img_copy = None
        
        votes_thresh = hough_votes_thresh_abs if hough_votes_thresh_abs else round(self.img_w * hough_votes_thresh_rel)
        
        # detect lines with hough transform
        lines = cv2.HoughLines(self.edges, hough_rho_res, hough_theta_res, votes_thresh)
        
        lines_hough, lines_ab = self._generate_hough_and_ab_lines(lines, self.img_w, self.img_h)
        
        if return_output_images:
            for p1, p2, line_dir in lines_ab:
                p1 = tuple(p1)
                p2 = tuple(p2)
                cv2.line(lines_only_img, p1, p2, 255, 1)
                line_color = (0, 255, 0) if line_dir == DIRECTION_HORIZONTAL else (0, 0, 255)
                cv2.line(input_img_copy, p1, p2, line_color, 1)
                
            return lines_hough, lines_ab, lines_only_img, input_img_copy
        else:
            return lines_hough, lines_ab
    
    def find_rotation_or_skew(self, rot_thresh, rot_same_dir_thresh):
        """
        Find page rotation or horizontal/vertical skew using detected lines in <lines>. The lines list must consist
        of arrays with the line rotation "theta" at array index 1 like the returned list from detect_lines().
        <rot_thresh> is the minimum threshold in radians for a rotation to be counted as such.
        <rot_same_dir_thresh> is the maximum threshold for the difference between horizontal and vertical line
        rotation.
        """
        if not self.lines_hough or not self.lines_ab:
            raise ValueError("no lines present. did you run detect_lines()?")
        
        # get the deviations
        
        hori_deviations = []   # deviation from unity vector in x-direction
        vert_deviations = []   # deviation from unity vector in y-direction
        
        for _, _, theta_norm, line_dir in self.lines_hough:                        
            if line_dir == DIRECTION_VERTICAL:
                deviation = -theta_norm
                if deviation < -PIHLF:
                    deviation += np.pi
            else:
                deviation = PIHLF - theta_norm
                
            assert abs(deviation) <= PI4TH
            vert_deviations.append(-deviation)

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

    @property
    def imgfile(self):
        return self._imgfile
        
    @imgfile.setter
    def imgfile(self, v):
        self._imgfile = v
        # reset
        self.input_img = None
        self.img_w = None
        self.img_h = None
        
        self._reset()
    
    def _load_imgfile(self):
        # reset
        self._reset()
        
        if self.input_img is not None:   # already loaded
            return
        
        self.input_img = cv2.imread(self.imgfile)
        if self.input_img is None:
            raise IOError("could not load file '%s'" % self.imgfile)

        self.img_h, self.img_w = self.input_img.shape[:2]
    
    @staticmethod
    def _normalize_angle(theta):
        if theta >= np.pi:
            theta_norm = theta - np.pi
        elif theta < -np.pi:
            theta_norm = theta + 2 * np.pi
        elif theta < 0:
            theta_norm = theta + np.pi
        else:
            theta_norm = theta
        
        assert 0 <= theta_norm < np.pi
        
        return theta_norm

    @classmethod
    def _generate_hough_and_ab_lines(cls, lines, img_w, img_h):
        lines_hough = []
        lines_ab = []
        for l in lines:
            rho, theta = l[0]
            theta_norm = cls._normalize_angle(theta)
                
            if abs(PIHLF - theta_norm) > PI4TH:  # vertical
                line_dir = DIRECTION_VERTICAL
            else:
                line_dir = DIRECTION_HORIZONTAL
            
            # calculate intersections with image dimension minima/maxima
            
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
                        
            x_miny = round(rho / cos_theta) if cos_theta != 0 else img_w
            y_minx = round(rho / sin_theta) if sin_theta != 0 else img_h
            x_maxy = round((rho - img_h * sin_theta) / cos_theta) if cos_theta != 0 else img_w
            y_maxx = round((rho - img_w * cos_theta) / sin_theta) if sin_theta != 0 else img_h
            
            if 0 <= y_minx < img_h:
                x1 = 0
                if 0 <= y_minx < img_h:
                    y1 = y_minx
                else:
                    y1 = y_maxx
            else:
                if 0 <= x_maxy < img_w:
                    x1 = x_maxy
                else:
                    x1 = x_miny
                y1 = img_h
                
            if 0 <= x_maxy < img_w:
                if 0 <= x_miny < img_w:
                    x2 = x_miny
                else:
                    x2 = x_maxy
                y2 = 0
            else:
                x2 = img_w
                if 0 <= y_maxx < img_h:
                    y2 = y_maxx
                else:
                    y2 = y_minx
            
            lines_hough.append((rho, theta, theta_norm, line_dir))
            lines_ab.append((pt(x1, y1, np.int), pt(x2, y2, np.int), line_dir))
        
        return lines_hough, lines_ab
