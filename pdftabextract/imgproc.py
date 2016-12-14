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


class ImageProc:    
    def __init__(self, imgfile):
        if not imgfile:
            raise ValueError("paramter 'imgfile' must be a non-empty, non-None string")
        
        self._imgfile = imgfile
        self.input_img = None
        self.img_w = None
        self.img_h = None
        
        self.gray_img = None
        self.edges = None     # edges detected by Canny algorithm
    

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
            lines_only_img = np.zeros((self.img_w, self.img_h, 1), np.uint8)
            input_img_copy = np.copy(self.input_img)
        else:
            lines_only_img = None
            input_img_copy = None
        
        votes_thresh = hough_votes_thresh_abs if hough_votes_thresh_abs else round(self.img_w * hough_votes_thresh_rel)
        
        # detect lines with hough transform
        lines = cv2.HoughLines(self.edges, hough_rho_res, hough_theta_res, votes_thresh)
        
        if return_output_images:        
            max_img_dim = max(self.img_w, self.img_h)
            
            for l in lines:
                rho, theta = l[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + max_img_dim*(-b))
                y1 = int(y0 + max_img_dim*(a))
                x2 = int(x0 - max_img_dim*(-b))
                y2 = int(y0 - max_img_dim*(a))
                
                cv2.line(lines_only_img, (x1, y1), (x2, y2), 255, 1)
                cv2.line(input_img_copy, (x1, y1), (x2, y2), (0,255,0), 1)
            
            return lines, lines_only_img, input_img_copy
        else:
            return lines
    
    @staticmethod
    def find_rotation_or_skew(lines, rot_thresh, rot_same_dir_thresh):
        """
        Find page rotation or horizontal/vertical skew using detected lines in <lines>. The lines list must consist
        of arrays with the line rotation "theta" at array index 1 like the returned list from detect_lines().
        <rot_thresh> is the minimum threshold in radians for a rotation to be counted as such.
        <rot_same_dir_thresh> is the maximum threshold for the difference between horizontal and vertical line
        rotation.
        """
        pihlf = np.pi / 2
        pi4th = np.pi / 4

        hori_deviations = []
        vert_deviations = []
        
        for l in lines:
            _, theta = l[0]
            
            if theta >= np.pi:
                theta_norm = theta - np.pi
            elif theta < -np.pi:
                theta_norm = theta + 2 * np.pi
            elif theta < 0:
                theta_norm = theta + np.pi
            else:
                theta_norm = theta
            
            assert 0 <= theta_norm < np.pi
            
            hori_deviation = pihlf - theta_norm
                
            if abs(hori_deviation) > pi4th:  # vertical
                deviation = hori_deviation - pihlf
                if deviation < -pihlf:
                    deviation += np.pi
                assert abs(deviation) <= pi4th
                vert_deviations.append(-deviation)
            else:
                assert abs(hori_deviation) <= pi4th
                hori_deviations.append(-hori_deviation)

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
        self.gray_img = None
        self.edges = None
    
        
    def _load_imgfile(self):
        # reset
        self.gray_img = None
        self.edges = None
        
        if self.input_img is not None:   # already loaded
            return
        
        self.input_img = cv2.imread(self.imgfile)
        if self.input_img is None:
            raise IOError("could not load file '%s'" % self.imgfile)

        self.img_w, self.img_h = self.input_img.shape[:2]
