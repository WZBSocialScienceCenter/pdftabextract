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


def vecdist(p1, p2):
    return np.linalg.norm(p2-p1)


def vecangle(v1, v2):
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return np.nan
    return math.acos(np.vdot(v1, v2) / denom)


def vecrotate(v, theta, about=np.array((0,0))):
    cth = math.cos(theta)
    sth = math.sin(theta)
        
    return pt(
        cth * v[0] - sth * v[1] + about[0] - cth * about[0] + sth * about[1],
        sth * v[0] + cth * v[1] + about[1] - sth * about[0] - cth * about[1]
    )


def line_from_points(p1, p2):
    """
    Return a and b for y = a*x + b
    """
    v1 = p2 - p1
    a = v1[1] / v1[0]
    b = (p2[0] * p1[1] - p1[0] * p2[1]) / v1[0]
    
    return a, b


def pointintersect(p1, p2, p3, p4, check_in_segm=True):
    a, b = line_from_points(p1, p2)
    c, d = line_from_points(p3, p4)
    
    x = (d - b) / (a - c)
    y = (a * d - b * c) / (a - c)
    
    range_xs = (
        (min(p1[0], p2[0]), max(p1[0], p2[0])),
        (min(p3[0], p4[0]), max(p3[0], p4[0])),
    )
    range_ys = (
        (min(p1[1], p2[1]), max(p1[1], p2[1])),
        (min(p3[1], p4[1]), max(p3[1], p4[1])),
    )
    
    if not check_in_segm or (check_in_segm and all((rx[0] <= x <= rx[1] for rx in range_xs)) \
            and all((ry[0] <= y <= ry[1] for ry in range_ys))):
        return pt(x, y)
    else:
        return None


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