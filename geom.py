# -*- coding: utf-8 -*-
"""
Common functions for geometric calculations

Created on Wed Jun  1 16:37:23 2016

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import math

import numpy as np


def pt(x, y):
    return np.array((x, y))


def vecdist(p1, p2):
    return np.linalg.norm(p2-p1)


def vecangle(v1, v2):
    return math.acos(np.vdot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


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
