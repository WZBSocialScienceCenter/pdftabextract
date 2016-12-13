# -*- coding: utf-8 -*-
"""
Common functions for geometric calculations

Created on Wed Jun  1 16:37:23 2016

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import math

import numpy as np


def pt(x, y, dtype=np.float):
    return np.array((x, y), dtype=dtype)


def ptdist(p1, p2):
    """distance between two points p1, p2"""
    return np.linalg.norm(p2-p1)


def vecangle(v1, v2):
    """angle between two vectors v1, v2 in radians"""
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return np.nan
    return math.acos(np.vdot(v1, v2) / denom)


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


def pointintersect(p1, p2, p3, p4, check_in_segm=True):
    """
    Check if two lines made from (p1, p2) and (p3, p4) respectively, intersect.
    
    If check_in_segm is True, will check that the line segments actually intersect,
    will calculate intersection of inifite lines.
    
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
        if overlap(p1[0], p2[0], p3[0], p4[0]) and overlap(p1[1], p2[1], p3[1], p4[1]):
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
    :param leftop: np.array from pt()
    :param rightbottom: np.array from pt()
    :return 2x2 np.array matrix, first row is lefttop, second row is rightbottom
    """
    assert lefttop.dtype == rightbottom.dtype        
    
    return np.array((lefttop, rightbottom), dtype=lefttop.dtype)


def rectcenter(r):
    w = r[1][0] - r[0][0]
    h = r[1][1] - r[0][1]
    
    return pt(r[0][0] + w / 2, r[0][1] + h / 2)


def rectcenter_dist(r1, r2):
    return vecdist(rectcenter(r1), rectcenter(r2))


def rectarea(r):
    return (r[1][0] - r[0][0]) * (r[1][1] - r[0][1])


def rectintersect(a, b, norm_intersect_area=None):
    assert a.dtype == b.dtype    
    assert norm_intersect_area in (None, 'a', 'b')
    
    a_a = rectarea(a)
    a_b = rectarea(b)

    assert a_a >= 0
    assert a_b >= 0
    
    max_a = min(a_a, a_b)
    
    d = np.empty(4, dtype=a.dtype)
    # x
    d[0] = b[1][0] - a[0][0]
    d[1] = a[1][0] - b[0][0]
    
    # y
    d[2] = b[1][1] - a[0][1]
    d[3] = a[1][1] - b[0][1]
    
    if np.sum(d >= 0) == 4:
        if norm_intersect_area == 'a':
            norm_with = a_a
        elif norm_intersect_area == 'b':
            norm_with = a_b
        else:
            norm_with = 1.0
            
        return min(max_a, np.min(np.abs(d[0:2])) * np.min(np.abs(d[2:4]))) / norm_with
    else:
        return None
