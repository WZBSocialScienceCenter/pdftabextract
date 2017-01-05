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

from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y, DIRECTION_HORIZONTAL, DIRECTION_VERTICAL
from pdftabextract.geom import normalize_angle, project_polarcoord_lines
from pdftabextract.clustering import zip_clusters_and_values

PIHLF = np.pi / 2
PI4TH = np.pi / 4


def pt_to_tuple(p):
    return (int(round(p[0])), int(round(p[1])))


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

        
    def detect_lines(self, canny_low_thresh, canny_high_thresh, canny_kernel_size,
                     hough_rho_res, hough_theta_res, hough_votes_thresh_rel,
                     hough_votes_thresh_abs=None,
                     gray_conversion=cv2.COLOR_BGR2GRAY):
        """
        Detect lines in input image using hough transform.
        """
        
        self.gray_img = cv2.cvtColor(self.input_img, gray_conversion)
        self.edges = cv2.Canny(self.gray_img, canny_low_thresh, canny_high_thresh, apertureSize=canny_kernel_size)
        
        votes_thresh = hough_votes_thresh_abs if hough_votes_thresh_abs else round(self.img_w * hough_votes_thresh_rel)
        
        # detect lines with hough transform
        lines = cv2.HoughLines(self.edges, hough_rho_res, hough_theta_res, votes_thresh)
        
        self.lines_hough = self._generate_hough_lines(lines)
        
        return self.lines_hough
        
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
        """
        From a list of lines <lines_hough> in polar coordinate space, generate lines in cartesian coordinate space
        from points A to B in image dimension space. A and B are at the respective opposite borders
        of the line projected into the image.
        Will return a list with tuples (A, B, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL).
        """
        
        projected = project_polarcoord_lines([l[:2] for l in lines_hough], self.img_w, self.img_h)
        return [(p1, p2, line_dir) for (p1, p2), (_, _, _, line_dir) in zip(projected, lines_hough)]
    
    def find_rotation_or_skew(self, rot_thresh, rot_same_dir_thresh, omit_on_rot_thresh=None):
        """
        Find page rotation or horizontal/vertical skew using detected lines in <lines>. The lines list must consist
        of arrays with the line rotation "theta" at array index 1 like the returned list from detect_lines().
        <rot_thresh> is the minimum threshold in radians for a rotation to be counted as such.
        <rot_same_dir_thresh> is the maximum threshold for the difference between horizontal and vertical line
        rotation.
        <omit_on_rot_thresh> is an optional threshold to filter out "stray" lines whose angle is too far apart from
        the median angle of all other lines that go in the same direction.
        """
        if not self.lines_hough:
            raise ValueError("no lines present. did you run detect_lines()?")
        
        # get the deviations
        
        hori_deviations = []   # deviation from unit vector in x-direction
        vert_deviations = []   # deviation from unit vector in y-direction
        
        lines_w_deviations = [] if omit_on_rot_thresh is not None else None
        
        for rho, theta, theta_norm, line_dir in self.lines_hough:                        
            if line_dir == DIRECTION_VERTICAL:
                deviation = -theta_norm
                if deviation < -PIHLF:
                    deviation += np.pi
                vert_deviations.append(-deviation)
            else:
                deviation = PIHLF - theta_norm
                hori_deviations.append(-deviation)
            
            if omit_on_rot_thresh is not None:
                lines_w_deviations.append((rho, theta, theta_norm, line_dir, -deviation))
            
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
        
        hori_rot_above_thresh = abs(median_hori_dev) > rot_thresh
        vert_rot_above_thresh = abs(median_vert_dev) > rot_thresh
        
        if omit_on_rot_thresh is not None:
            assert len(lines_w_deviations) == len(self.lines_hough)
            lines_filtered = []
            for rho, theta, theta_norm, line_dir, deviation in lines_w_deviations:
                dir_dev = median_hori_dev if line_dir == DIRECTION_HORIZONTAL else median_vert_dev
                if abs(abs(dir_dev) - abs(deviation)) < omit_on_rot_thresh:
                    lines_filtered.append((rho, theta, theta_norm, line_dir))
            assert len(lines_filtered) <= len(self.lines_hough)
            self.lines_hough = lines_filtered
        
        if hori_rot_above_thresh and vert_rot_above_thresh:
            if abs(median_hori_dev - median_vert_dev) < rot_same_dir_thresh:
                return ROTATION, (median_hori_dev + median_vert_dev) / 2
            else:
                warning('horizontal / vertical rotation not in same direction (%f / %f)'
                      % (degrees(median_hori_dev), degrees(median_vert_dev)))
        elif hori_rot_above_thresh:
            return SKEW_Y, median_hori_dev
        elif vert_rot_above_thresh:
            return SKEW_X, median_vert_dev

        return None, None
    
    def find_clusters(self, direction, method, **kwargs):
        remove_empty_cluster_sections_use_texts = kwargs.pop('remove_empty_cluster_sections_use_texts', None)
        if remove_empty_cluster_sections_use_texts is not None:
            remove_empty_cluster_sections_n_texts_ratio = kwargs.pop('remove_empty_cluster_sections_n_texts_ratio')
            remove_empty_cluster_sections_scaling = kwargs.pop('remove_empty_cluster_sections_scaling')
            remove_empty_cluster_sections_clust_center_fn = kwargs.pop('remove_empty_cluster_sections_clust_center_fn',
                                                                       np.median)
        
        if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
            raise ValueError("invalid value for 'direction': '%s'" % direction)
        
        if not callable(method):
            raise ValueError("'method' must be callable")
        
        lines_ab = self.ab_lines_from_hough_lines([l for l in self.lines_hough if l[3] == direction])
        
        coord_idx = 0 if direction == DIRECTION_VERTICAL else 1
        positions = np.array([(l[0][coord_idx] + l[1][coord_idx]) / 2 for l in lines_ab])
        
        clusters = method(positions, **kwargs)
        
        if type(clusters) != list:
            raise ValueError("'method' returned invalid clusters (must be list)")
        
        if len(clusters) > 0 and type(clusters[0]) != np.ndarray:
            raise ValueError("'method' returned invalid cluster elements (must be list of numpy.ndarray objects)")
        
        clusters_w_vals = zip_clusters_and_values(clusters, positions)
        
        if remove_empty_cluster_sections_use_texts is not None:
            if direction == DIRECTION_HORIZONTAL:
                t_attr = 'top', 'bottom'
            else:
                t_attr = 'left', 'right'
                
            clusters_w_vals_and_centers = [(ind, vals, remove_empty_cluster_sections_clust_center_fn(vals) / remove_empty_cluster_sections_scaling)
                                           for ind, vals in clusters_w_vals]
            clusters_w_vals_and_centers = sorted(clusters_w_vals_and_centers, key=lambda x: x[2])
            clusters_w_vals_and_n_texts = []
            prev_center = -1
            for ind, vals, center in clusters_w_vals_and_centers:
                texts = [t for t in remove_empty_cluster_sections_use_texts
                         if prev_center < t[t_attr[0]] <= center or prev_center < t[t_attr[1]] <= center]
                clusters_w_vals_and_n_texts.append((ind, vals, len(texts)))
                prev_center = center
            assert len(clusters_w_vals_and_n_texts) == len(clusters_w_vals)
                
            max_n_texts = np.median([tup[2] for tup in clusters_w_vals_and_n_texts])
            n_texts_thresh = round(max_n_texts * remove_empty_cluster_sections_n_texts_ratio)
            clusters_w_vals_filtered = []
            prev_clust = None
            for ind, vals, n_texts in clusters_w_vals_and_n_texts:
                if n_texts >= n_texts_thresh:
                    if not clusters_w_vals_filtered and prev_clust is not None:
                        clusters_w_vals_filtered.append(prev_clust)
                    
                    clusters_w_vals_filtered.append((ind, vals))
                prev_clust = (ind, vals)
            assert(len(clusters_w_vals_filtered) <= len(clusters_w_vals))
            clusters_w_vals = clusters_w_vals_filtered
        
        return clusters_w_vals
            
    def draw_lines(self, orig_img_as_background=True):
        lines_ab = self.ab_lines_from_hough_lines(self.lines_hough)
        
        baseimg = self._baseimg_for_drawing(orig_img_as_background)
        
        for p1, p2, line_dir in lines_ab:
            line_color = (0, 255, 0) if line_dir == DIRECTION_HORIZONTAL else (0, 0, 255)
            
            cv2.line(baseimg, pt_to_tuple(p1), pt_to_tuple(p2), line_color, self.DRAW_LINE_WIDTH)
        
        return baseimg

    def draw_line_clusters(self, direction, clusters_w_vals, orig_img_as_background=True):
        if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
            raise ValueError("invalid value for 'direction': '%s'" % direction)
        
        baseimg = self._baseimg_for_drawing(orig_img_as_background)
        
        n_colors = len(clusters_w_vals)
        color_incr = max(1, round(255 / n_colors))
        
        for i, (_, vals) in enumerate(clusters_w_vals):
            i += 2
            
            line_color = (
                (color_incr * i) % 256,
                (color_incr * i * i) % 256,
                (color_incr * i * i * i) % 256,
            )
            
            self.draw_lines_in_dir(baseimg, direction, vals, line_color)
        
        return baseimg
    
    @staticmethod
    def draw_lines_in_dir(baseimg, direction, line_positions, line_color, line_width=None):
        if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
            raise ValueError("invalid value for 'direction': '%s'" % direction)
        
        if not line_width:
            line_width = ImageProc.DRAW_LINE_WIDTH
    
        h, w = baseimg.shape[:2]
        
        for pos in line_positions:
            pos = int(round(pos))
            
            if direction == DIRECTION_HORIZONTAL:
                p1 = (0, pos)
                p2 = (w, pos)
            else:
                p1 = (pos, 0)
                p2 = (pos, h)
            
            cv2.line(baseimg, p1, p2, line_color, line_width)
    
    def _baseimg_for_drawing(self, use_orig):
        if use_orig:
            return np.copy(self.input_img)
        else:
            return np.zeros((self.img_h, self.img_w, 3), np.uint8)
        
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
