# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:50:51 2017

@author: mkonrad
"""

import math

import pytest
import numpy as np

from pdftabextract.geom import pt, ptdist, vecangle, vecrotate, overlap, lineintersect, rect


def test_pt():
    x = 0
    y = 1
    pt0 = pt(x, y)
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
