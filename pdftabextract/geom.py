# -*- coding: utf-8 -*-
"""
Common functions for geometric calculations

Created on Wed Jun  1 16:37:23 2016

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import math

import numpy as np


def pt(x, y, dtype=np.float):
    """Create a point in 2D space at <x>, <y>"""
    return np.array((x, y), dtype=dtype)


def ptdist(p1, p2):
    """distance between two points p1, p2"""
    return np.linalg.norm(p2-p1)


def vecangle(v1, v2):
    """angle between two vectors v1, v2 in radians"""
    v0 = pt(0, 0)
        
    if np.allclose(v1, v0) or np.allclose(v2, v0):
        return np.nan
    
    if np.allclose(v1, v2):
        return 0
    
    num = np.vdot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    if np.isclose(num, denom):
        return 0
    
    return math.acos(num / denom)


def vecrotate(v, theta, about=np.array((0,0))):
    """rotate a vector v by angle theta (in radians) about point <about>"""
    cth = math.cos(theta)
    sth = math.sin(theta)

    return pt(
        cth * v[0] - sth * v[1] + about[0] - cth * about[0] + sth * about[1],
        sth * v[0] + cth * v[1] + about[1] - sth * about[0] - cth * about[1]
    )


def overlap(a1, a2, b1, b2):
    """
    Check if ranges a1-a2 and b1-b2 overlap.
    """
    a_min = min(a1, a2)
    a_max = max(a1, a2)
    b_min = min(b1, b2)
    b_max = max(b1, b2)
    
    return a_min <= b_min <= a_max or b_min <= a_min <= b_max or \
           a_min <= b_max <= a_max or b_min <= a_max <= b_max


def lineintersect(p1, p2, p3, p4, check_in_segm=True):
    """
    Check if two lines made from (p1, p2) and (p3, p4) respectively, intersect.
    
    If check_in_segm is True, will check that the line segments actually intersect,
    otherwise will calculate intersection of inifite lines.
    
    If no intersection is found, returns None
    For parallel lines, returns pt(np.nan, np.nan) if they are coincident.
    For non-parallel lines, returns point of intersection as pt(x, y)
    
    See http://mathworld.wolfram.com/Line-LineIntersection.html
    """    
    
    # check for intersection in infinity
    p1p2 = np.array([p1, p2])
    p3p4 = np.array([p3, p4])
    
    det_p1p2 = np.linalg.det(p1p2)
    det_p3p4 = np.linalg.det(p3p4)
    diff_x12 = p1[0] - p2[0]
    diff_x34 = p3[0] - p4[0]
    diff_y12 = p1[1] - p2[1]
    diff_y34 = p3[1] - p4[1]
    
    x_num_mat = np.array([[det_p1p2, diff_x12], [det_p3p4, diff_x34]])
    y_num_mat = np.array([[det_p1p2, diff_y12], [det_p3p4, diff_y34]])
    den_mat = np.array([[diff_x12, diff_y12], [diff_x34, diff_y34]])
    den = np.linalg.det(den_mat)
    
    if den == 0:  # parallel
        isect_x = np.nan
        isect_y = np.nan
        parallel = True
    else:         # not parallel
        isect_x = np.linalg.det(x_num_mat) / den
        isect_y = np.linalg.det(y_num_mat) / den
        parallel = False
    
    P = pt(isect_x, isect_y)    
    
    if not check_in_segm and not parallel:   # no segment checking necessary
        return P
        
    if parallel:  # check if parallel segments are coincident
        if not check_in_segm or (overlap(p1[0], p2[0], p3[0], p4[0]) and overlap(p1[1], p2[1], p3[1], p4[1])):
            return P      # lines coincident -> return pt(np.nan, np.nan)
        else:
            return None   # no intersection in segment, only parallel
    else:  # non parallel intersection -> check if segments are in range
        range_xs = (
            (min(p1[0], p2[0]), max(p1[0], p2[0])),
            (min(p3[0], p4[0]), max(p3[0], p4[0])),
        )
        range_ys = (
            (min(p1[1], p2[1]), max(p1[1], p2[1])),
            (min(p3[1], p4[1]), max(p3[1], p4[1])),
        )        
        
        in_range = (all((rx[0] <= P[0] <= rx[1] for rx in range_xs)) \
                    and all((ry[0] <= P[1] <= ry[1] for ry in range_ys)))
        if in_range:
            return P     # intersection of segments
        else:
            return None  # no intersection


def rect(lefttop, rightbottom):
    """
    create a rectangle structure from top left and bottom right point.
    :param leftop: np.array from pt()
    :param rightbottom: np.array from pt()
    :return 2x2 np.array matrix, first row is top left point, second row is bottom right point
    """
    if lefttop.dtype != rightbottom.dtype:
        raise ValueError('dtypes of lefttop and rightbottom must match')
    
    if lefttop[0] >= rightbottom[0]:
        raise ValueError('lefttop[0] must be smaller than rightbottom[0] to form a rectangle')
    
    if lefttop[1] >= rightbottom[1]:
        raise ValueError('lefttop[1] must be smaller than rightbottom[1] to form a rectangle')
    
    return np.array((lefttop, rightbottom), dtype=lefttop.dtype)

    
def rect_from_text(t):
    """create a rectangle from a text box <t>"""
    return rect(t['topleft'], t['bottomright'])

    
def rectcenter(r):
    """"Return the center of a rectangle <r>."""
    w = r[1][0] - r[0][0]
    h = r[1][1] - r[0][1]
    
    return pt(r[0][0] + w / 2, r[0][1] + h / 2)


def rectcenter_dist(r1, r2):
    """Return the distance between centers of rectangles <r1> and <r2>"""
    return ptdist(rectcenter(r1), rectcenter(r2))


def rectarea(r):
    """Return the area of rectangle <r>"""
    return (r[1][0] - r[0][0]) * (r[1][1] - r[0][1])


def rectintersect(a, b, norm_intersect_area=None):
    """
    Check for rectangle intersection between rectangles <a> and <b>.
    Return None if no intersection, else return the area of the intersection, optionally normalized/scaled to
    <norm_intersect_area>.
    """
    if a.dtype != b.dtype:
        raise ValueError('dtypes of a and b must match')
    
    if norm_intersect_area not in (None, 'a', 'b'):
        raise ValueError("norm_intersect_area must be None, 'a' or 'b'")
    
    a_a = rectarea(a)
    a_b = rectarea(b)

    if a_a <= 0:
        raise ValueError('Area of a must be > 0')
    if a_b <= 0:
        raise ValueError('Area of b must be > 0')
    
    max_a = min(a_a, a_b)
    
    # deltas per axis
    d = np.empty(4, dtype=a.dtype)
    
    # x
    d[0] = b[1][0] - a[0][0]
    d[1] = a[1][0] - b[0][0]
    
    # y
    d[2] = b[1][1] - a[0][1]
    d[3] = a[1][1] - b[0][1]
    
    if np.sum(d >= 0) == 4:  # intersection
        if norm_intersect_area == 'a':
            norm_with = a_a
        elif norm_intersect_area == 'b':
            norm_with = a_b
        else:
            norm_with = 1.0
            
        return min(max_a, np.min(np.abs(d[0:2])) * np.min(np.abs(d[2:4]))) / norm_with
    else:  # no intersection
        return None


def normalize_angle(theta):
    """Normalize an angle theta to theta_norm so that: 0 <= theta_norm < 2 * np.pi"""
    twopi = 2 * np.pi
    
    if theta >= twopi:
        m = math.floor(theta/twopi)
        if theta/twopi - m > 0.99999:   # account for rounding errors
            m += 1        
        theta_norm = theta - m * twopi
    elif theta < 0:
        m = math.ceil(theta/twopi)
        if theta/twopi - m < -0.99999:   # account for rounding errors
            m -= 1
        theta_norm = abs(theta - m * twopi)
    else:
        theta_norm = theta
    
    return theta_norm


def normalize_angle_halfcircle(theta):
    theta_norm = normalize_angle(theta)
    return theta_norm if theta_norm < np.pi else theta_norm - np.pi


def project_polarcoord_lines(lines, img_w, img_h):
    """
    Project lines in polar coordinate space <lines> (e.g. from hough transform) onto a canvas of size
    <img_w> by <img_h>.
    """
    
    if img_w <= 0:
        raise ValueError('img_w must be > 0')
    if img_h <= 0:
        raise ValueError('img_h must be > 0')
    
    lines_ab = []
    for i, (rho, theta) in enumerate(lines):
        # calculate intersections with canvas dimension minima/maxima
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
                    
        x_miny = rho / cos_theta if cos_theta != 0 else float("inf")  # x for a minimal y (y=0)
        y_minx = rho / sin_theta if sin_theta != 0 else float("inf")  # y for a minimal x (x=0)
        x_maxy = (rho - img_w * sin_theta) / cos_theta if cos_theta != 0 else float("inf")  # x for maximal y (y=img_h)
        y_maxx = (rho - img_h * cos_theta) / sin_theta if sin_theta != 0 else float("inf")  # y for maximal x (y=img_w)
        
        # because rounding errors happen, sometimes a point is counted as invalid because it
        # is slightly out of the bounding box
        # this is why we have to correct it like this
        
        def border_dist(v, border):
            return v if v <= 0 else v - border
        
        # set the possible points
        # some of them will be out of canvas
        possible_pts = [
            ([x_miny, 0], (border_dist(x_miny, img_w), 0)),
            ([0, y_minx], (border_dist(y_minx, img_h), 1)),
            ([x_maxy, img_h], (border_dist(x_maxy, img_w), 0)),
            ([img_w, y_maxx], (border_dist(y_maxx, img_h), 1)),
        ]
        
        # get the valid and the dismissed (out of canvas) points
        valid_pts = []
        dismissed_pts = []
        for p, dist in possible_pts:
            if 0 <= p[0] <= img_w and 0 <= p[1] <= img_h:
                valid_pts.append(p)
            else:
                dismissed_pts.append((p, dist))
        
        # from the dismissed points, get the needed ones that are closed to the canvas       
        n_needed_pts = 2 - len(valid_pts)
        if n_needed_pts > 0:
            dismissed_pts_sorted = sorted(dismissed_pts, key=lambda x: abs(x[1][0]), reverse=True)

            for _ in range(n_needed_pts):
                p, (dist, coord_idx) = dismissed_pts_sorted.pop()
                p[coord_idx] -= dist  # correct
                valid_pts.append(p)
        
        
        p1 = pt(*valid_pts[0])
        p2 = pt(*valid_pts[1])
        
        lines_ab.append((p1, p2))
    
    return lines_ab

