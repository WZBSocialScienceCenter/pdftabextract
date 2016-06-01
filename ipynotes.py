# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:11:33 2016

@author: mkonrad
"""

import xml.etree.ElementTree as ET
import math
from copy import copy
from logging import warn

import numpy as np

#%%

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5

LEFTMOST_COL_ALIGN = 'topleft'     # topleft, topright or center
RIGHTMOST_COL_ALIGN = 'topleft'    # topleft, topright or center

MAX_PAGE_ROTATION_RANGE = math.radians(1.0)     # 1 degree

#%%

tree = ET.parse('testxmls/1992_93.pdf.xml')
root = tree.getroot()

#%%
def sorted_by_attr(texts, attr):
    return sorted(texts, key=lambda x: x[attr])

def parse_pages(root):
    pages = {}
        
    for p in root.findall('page'):
        p_num = int(p.attrib['number'])
        page = {
            'number': p_num,
            'width': int(p.attrib['width']),
            'height': int(p.attrib['height']),
            'texts': [],
            'xmlnode': p
        }
        
        for t in p.findall('text'):
            if not t.text:  # filter out text elements without content
                continue
            
            t_top = int(t.attrib['top'])
            t_left = int(t.attrib['left'])
            t_width = int(t.attrib['width'])
            t_height = int(t.attrib['height'])
            
            t_bottom = t_top + t_height
            t_right = t_left + t_width
    
            text = {
                'top': t_top,
                'left': t_left,
                'bottom': t_bottom,
                'right': t_right,
                'width': t_width,
                'height': t_height,
                'center': np.array((t_left + t_width / 2, t_top + t_height / 2)),
                'topleft': np.array((t_left, t_top)),
                'bottomleft': np.array((t_left, t_bottom)),
                'topright': np.array((t_right, t_top)),
                'bottomright': np.array((t_right, t_bottom)),
                'value': t.text,  # only for easy identification during debugging. TODO: delete
                'xmlnode': t
            }
            
            page['texts'].append(text)
            
        # page['texts_leftright'] = sorted_by_attr(page['texts'], 'center_x')
        # page['texts_topdown'] = sorted_by_attr(page['texts'], 'center_y')
        pages[p_num] = page

    return pages


def get_bodytexts(page):
    miny = page['height'] * HEADER_RATIO
    maxy = page['height'] * (1 - FOOTER_RATIO)
    
    return list(filter(lambda t: t['top'] >= miny and t['bottom'] <= maxy, page['texts']))    


def divide_texts_horizontally(page, texts=None):
    if texts is None:
        texts = page['texts']
    
    divide_x = page['width'] * DIVIDE_RATIO
    lefttexts = list(filter(lambda t: t['right'] <= divide_x, texts))
    righttexts = list(filter(lambda t: t['right'] > divide_x, texts))
    
    assert len(lefttexts) + len(righttexts) == len(texts)
    
    subpage_tpl = {
        'number': page['number'],
        'width': divide_x,
        'height': page['height'],
        'parentpage': page
    }
    
    subpage_left = copy(subpage_tpl)
    subpage_left['subpage'] = 'left'
    subpage_left['x_offset'] = 0
    subpage_left['texts'] = lefttexts
    
    subpage_right = copy(subpage_tpl)
    subpage_right['subpage'] = 'right'
    subpage_right['x_offset'] = divide_x
    subpage_right['texts'] = righttexts
    
    return subpage_left, subpage_right


def mindist_text(texts, origin, pos_attr):
    texts_by_dist = sorted(texts, key=lambda t: vecdist(origin, t[pos_attr]))
    return texts_by_dist[0]


def texts_at_page_corners(p):
    """
    :param p page or subpage
    """
    # TODO: offset beachten für sub_right!
    
    text_topleft = mindist_text(p['texts'], (0, 0), 'topleft')
    text_bottomleft = mindist_text(p['texts'], (0, p['height']), 'bottomleft')
    text_topright = mindist_text(p['texts'], (p['width'], 0), 'topright')
    text_bottomright = mindist_text(p['texts'], (p['width'], p['height']), 'bottomright')
    
    return text_topleft, text_topright, text_bottomright, text_bottomleft



def page_rotation_angle(text_topleft, text_topright, text_bottomright, text_bottomleft):
    vec_left = text_bottomleft[LEFTMOST_COL_ALIGN] - text_topleft[LEFTMOST_COL_ALIGN]
    vec_right = text_bottomright[LEFTMOST_COL_ALIGN] - text_topright[LEFTMOST_COL_ALIGN]
    vec_top = text_topright[LEFTMOST_COL_ALIGN] - text_topleft[LEFTMOST_COL_ALIGN]
    vec_bottom = text_bottomright[LEFTMOST_COL_ALIGN] - text_bottomleft[LEFTMOST_COL_ALIGN]
    
    up = np.array((0, 1))
    right = np.array((1, 0))
    
    left_angle = vecangle(vec_left, up)
    right_angle = vecangle(vec_right, up)
    top_angle = vecangle(vec_top, right)
    bottom_angle = vecangle(vec_bottom, right)
    
    angles = np.array((left_angle, right_angle, top_angle, bottom_angle))
    rng = np.max(angles) - np.min(angles)
    
    if rng > MAX_PAGE_ROTATION_RANGE:
        warn('big range of values for page rotation estimation: %f degrees' % math.degrees(rng))
    
    return np.median(angles)    


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

#%%
pages = parse_pages(root)

page = pages[35]

bodytexts = get_bodytexts(page)
subpages = divide_texts_horizontally(page, bodytexts)

#%%

sub_left = subpages[0]
page_corners_texts = texts_at_page_corners(sub_left)
page_rot = page_rotation_angle(*page_corners_texts)

text_topleft, text_bottomleft = page_corners_texts[0], page_corners_texts[3]
bottomline_pts = (
    pt(0, sub_left['height']),
    pt(sub_left['width'], sub_left['height'])
)
rot_about = pointintersect(text_topleft['topleft'], text_bottomleft['bottomleft'], *bottomline_pts, check_in_segm=False)
rot_about

#%%

for t in sub_left['texts']:
    t_pt = pt(t['left'], t['top'])
    t_pt_rot = vecrotate(t_pt, -page_rot, rot_about)
    t['xmlnode'].attrib['left'] = str(round(t_pt_rot[0]))
    t['xmlnode'].attrib['top'] = str(round(t_pt_rot[1]))

#%%

tree.write('1992_93_rotback.pdf.xml')