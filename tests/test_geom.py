# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:50:51 2017

@author: mkonrad
"""

import math

import pytest
from hypothesis import given
import hypothesis.strategies as st 
import numpy as np

from pdftabextract.geom import (pt, ptdist, vecangle, vecrotate, overlap, lineintersect,
                                rect, rectcenter, rectarea, rectintersect,
                                normalize_angle, normalize_angle_halfcircle,
                                project_polarcoord_lines)

FMIN = np.finfo(np.float32).min
FMAX = np.finfo(np.float32).max

               
def test_pt():
    x = 0
    y = 1
    pt0 = pt(x, y)
    assert type(pt0) is np.ndarray
    assert pt0.dtype == np.float
    assert pt0[0] == x
    assert pt0[1] == y

    pt1 = pt(x, y, np.int)
    assert pt1.dtype == np.int
    assert pt1[0] == x
    assert pt1[1] == y


def test_ptdist():
    p1 = pt(0, 0)
    p2 = pt(1, 0)
    p3 = pt(1, 1)
    
    assert ptdist(p1, p1) == 0
    assert ptdist(p1, p2) == 1
    assert ptdist(p2, p1) == ptdist(p1, p2)

    assert ptdist(p1, p3) == math.sqrt(2)
    

def test_vecangle():
    v1 = pt(1, 0)
    v2 = pt(2, 0)
    v3 = pt(1, 1)
    v4 = pt(0, 1)
    v5 = pt(0, -1)
    
    assert np.isnan(vecangle(pt(0, 0), v1))   # pt(0, 0) is vec of no length
    assert vecangle(v1, v2) == 0
    assert round(vecangle(v1, v3), 4) == round(math.radians(45), 4)
    assert vecangle(v2, v4) == vecangle(v1, v4) == math.radians(90)
    assert vecangle(v2, v5) == math.radians(90)   # always the smaller angle


@given(st.floats(min_value=FMIN, max_value=FMAX),
       st.floats(min_value=FMIN, max_value=FMAX),
       st.floats(min_value=FMIN, max_value=FMAX),
       st.floats(min_value=FMIN, max_value=FMAX))
def test_vecangle_2(x1, y1, x2, y2):
    v0 = pt(0, 0)
    v1 = pt(x1, y1)
    v2 = pt(x2, y2)
    
    try:
        alpha = vecangle(v1, v2)
    except ValueError:   # math domain error in some edge cases?
        return
    
    if np.allclose(v1, v0) or np.allclose(v2, v0):
        assert np.isnan(alpha)
    else:
        assert 0 <= alpha <= np.pi


def test_vecrotate():
    assert np.array_equal(vecrotate(pt(0, 0), 0.123), pt(0, 0))
    assert np.allclose(vecrotate(pt(1, 0), math.radians(90)), pt(0, 1))
    assert np.allclose(vecrotate(pt(1, 0), math.radians(90), about=pt(1, 1)), pt(2, 1))
    

def test_overlap():
    assert overlap(0, 1, 0, 1) is True
    assert overlap(0, 0, 1, 1) is False
    assert overlap(0, 10, 5, 15) is True
    assert overlap(-10, 10, -20, -10) is True
    assert overlap(-9, 10, -20, -10) is False


def test_lineintersect():
    # first with check_in_segm = True
    X = lineintersect(pt(0, 0), pt(0, 0), pt(0, 0), pt(0, 0))   # coincident I
    assert sum(np.isnan(X)) == len(X)
    
    X = lineintersect(pt(0, 0), pt(0, 1), pt(0, 0), pt(0, 1))   # coincident II
    assert sum(np.isnan(X)) == len(X)
    
    assert lineintersect(pt(0, 0), pt(0, 1), pt(1, 0), pt(1, 1)) is None  # parallel, non coincident
    assert lineintersect(pt(0, 0), pt(0, 1), pt(1, 1), pt(2, 2)) is None  # non-parellel, no intersection
    assert lineintersect(pt(0, 0), pt(2, 2), pt(0, 5), pt(5, 0)) is None  # non-parellel, no intersection II
    assert np.array_equal(lineintersect(pt(0, 0), pt(0, 1), pt(0, 1), pt(2, 2)), pt(0, 1))  # intersection - touch
    assert np.array_equal(lineintersect(pt(0, 0), pt(2, 2), pt(0, 2), pt(2, 0)), pt(1, 1))  # intersection

    # now with check_in_segm = False
    X = lineintersect(pt(0, 0), pt(0, 0), pt(0, 0), pt(0, 0), False)   # coincident I
    assert sum(np.isnan(X)) == len(X)
    
    X = lineintersect(pt(0, 0), pt(0, 1), pt(0, 0), pt(0, 1), False)   # coincident II
    assert sum(np.isnan(X)) == len(X)

    X = lineintersect(pt(0, 0), pt(1, 1), pt(2, 2), pt(3, 3), False)   # coincident III
    assert sum(np.isnan(X)) == len(X)
    
    assert np.array_equal(lineintersect(pt(0, 0), pt(0, 1), pt(1, 1), pt(2, 2), False), pt(0, 0))  # intersection (out of segments)
    assert np.array_equal(lineintersect(pt(0, 0), pt(0, 1), pt(0, 1), pt(2, 2), False), pt(0, 1))  # intersection - touch
    assert np.array_equal(lineintersect(pt(0, 0), pt(2, 2), pt(0, 2), pt(2, 0), False), pt(1, 1))  # intersection


def test_rect():
    with pytest.raises(ValueError):
        rect(pt(0, 0), pt(1, 1, dtype=np.int))  # dtypes do not match
    
    with pytest.raises(ValueError):
        rect(pt(0, 0), pt(0, 0))  # doesn't form rect

    with pytest.raises(ValueError):
        rect(pt(1, 1), pt(0, 0))  # doesn't form rect

    with pytest.raises(ValueError):
        rect(pt(0, 0), pt(1, 0))  # doesn't form rect
    
    a = pt(0, 0)
    b = pt(1, 1)
    r = rect(a, b)
    assert r.dtype == a.dtype == b.dtype
    assert np.array_equal(r[0], a)
    assert np.array_equal(r[1], b)
    
    a = pt(-3, -1)
    b = pt(8, 1.2)
    r = rect(a, b)
    assert r.dtype == a.dtype == b.dtype
    assert np.array_equal(r[0], a)
    assert np.array_equal(r[1], b)


def test_rectcenter():
    a = pt(0, 0)
    b = pt(1, 1)
    r = rect(a, b)
    center = rectcenter(r)
    assert type(center) is np.ndarray
    assert np.array_equal(center, pt(0.5, 0.5))
    
    a = pt(-3, -1)
    b = pt(2, 5)
    r = rect(a, b)
    assert np.array_equal(rectcenter(r), pt(-0.5, 2))


def test_rectarea():
    a = pt(0, 0)
    b = pt(1, 1)
    r = rect(a, b)
    assert rectarea(r) == 1
                   
    a = pt(-3, -1)
    b = pt(2, 5)
    r = rect(a, b)
    assert rectarea(r) == 30


def test_rectintersect():
    a = rect(pt(0, 0), pt(1, 1))
    b = rect(pt(-3, -1), pt(2, 5))
    
    assert rectintersect(a, a) == rectarea(a)
    assert rectintersect(b, b) == rectarea(b)
    assert rectintersect(a, a, norm_intersect_area='a') == 1
    assert rectintersect(a, a, norm_intersect_area='b') == 1
                        
    with pytest.raises(ValueError):
        rectintersect(a, a, norm_intersect_area='c')
    
    # complete intersect
    assert rectintersect(a, b) == rectarea(a)
    assert rectintersect(b, a) == rectarea(a)
    assert rectintersect(a, b, norm_intersect_area='a') == 1
    assert rectintersect(b, a, norm_intersect_area='b') == 1
    assert rectintersect(b, a, norm_intersect_area='a') < 1
    assert rectintersect(a, b, norm_intersect_area='b') < 1

    # partial intersect
    a = rect(pt(0, 0), pt(1, 1))
    b = rect(pt(0.5, 0.5), pt(1.5, 1.5))
    assert rectintersect(a, b) == 0.25
    assert rectintersect(a, b, norm_intersect_area='a') == 0.25
    assert rectintersect(a, b, norm_intersect_area='b') == 0.25
    b = rect(pt(0.75, 0.5), pt(1.5, 1.5))
    assert rectintersect(a, b) == 0.125

    # touch
    a = rect(pt(0, 0), pt(1, 1))
    b = rect(pt(1, 1), pt(1.5, 1.5))
    assert rectintersect(a, b) == 0

    # no intersection
    a = rect(pt(0, 0), pt(1, 1))
    b = rect(pt(1.1, 1.1), pt(1.5, 1.5))
    assert rectintersect(a, b) is None


def test_normalize_angle():
    for i in range(-10, 10):
        theta = i * np.pi
        norm = normalize_angle(theta)
        assert 0 <= norm < 2 * np.pi
        assert norm / np.pi == i % 2


def test_normalize_angle_halfcircle():
    for i in range(-10, 10):
        theta = 0.5 * i * np.pi
        norm = normalize_angle_halfcircle(theta)
        assert 0 <= norm < np.pi
        assert norm / np.pi * 2 == i % 2

@given(
    st.lists(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=2)),
    st.integers(),
    st.integers()
)
def test_project_polarcoord_lines(hough_lines, img_w, img_h):
    if img_w <= 0 or img_h <= 0:
        with pytest.raises(ValueError):
            project_polarcoord_lines(hough_lines, img_w, img_h)
        return
    else:
        res = project_polarcoord_lines(hough_lines, img_w, img_h)
    assert type(res) is list
    assert len(res) == len(hough_lines)
    for pts in res:
        assert len(pts) == 2
        assert type(pts[0]) == type(pts[1]) == np.ndarray
        assert len(pts[0]) == len(pts[1]) == 2

