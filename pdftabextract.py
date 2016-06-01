# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:40:39 2016

@author: mkonrad
"""

import xml.etree.ElementTree as ET
import math
from copy import copy
from logging import info, warning, error

import numpy as np

from geom import pt, vecangle, vecdist, vecrotate, pointintersect

#%%

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
#DIVIDE_RATIO = 0.5
DIVIDE_RATIO = None

LEFTMOST_COL_ALIGN = 'topleft'     # topleft, topright or center
RIGHTMOST_COL_ALIGN = 'topleft'    # topleft, topright or center

MAX_PAGE_ROTATION_RANGE = math.radians(1.0)     # 1 degree


#%%
def main():
    tree, root = read_xml('testxmls/1992_93.pdf.xml')
    
    pages = parse_pages(root)
    page = pages[35]

    bodytexts = get_bodytexts(page)  # strip off footer and header
    if DIVIDE_RATIO:
        subpages = divide_texts_horizontally(page, bodytexts)
    else:
        subpages = (page, )

    for sub_p in subpages:
        fix_rotation(sub_p)
    
    tree.write('1992_93_rotback.pdf.xml')


#%%

def read_xml(fname):
    tree = ET.parse(fname)
    root = tree.getroot()
    
    return tree, root

#%%

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
            
            page['texts'].append(create_text_dict(t))

        pages[p_num] = page

    return pages


def create_text_dict(t):
    t_width = int(t.attrib['width'])
    t_height = int(t.attrib['height'])

    text = {
        'width': t_width,
        'height': t_height,
        'value': t.text,  # only for easy identification during debugging. TODO: delete
        'xmlnode': t
    }
    
    update_text_dict_pos(text, pt(int(t.attrib['left']), int(t.attrib['top'])))
    
    return text


def update_text_dict_pos(t, pos, update_node=False):
    t_top = pos[1]
    t_left = pos[0]
    t_bottom = t_top + t['height']
    t_right = t_left + t['width']
    
    t.update({
        'top': t_top,
        'left': t_left,
        'bottom': t_bottom,
        'right': t_right,
        'topleft': np.array((t_left, t_top)),
        'bottomleft': np.array((t_left, t_bottom)),
        'topright': np.array((t_right, t_top)),
        'bottomright': np.array((t_right, t_bottom)),
    })

    if update_node:    
        t['xmlnode'].attrib['left'] = str(round(pos[0]))
        t['xmlnode'].attrib['top'] = str(round(pos[1]))


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
    # TODO: offset beachten für sub_right!
    vec_left = text_bottomleft[LEFTMOST_COL_ALIGN] - text_topleft[LEFTMOST_COL_ALIGN]
    vec_right = text_bottomright[LEFTMOST_COL_ALIGN] - text_topright[LEFTMOST_COL_ALIGN]
    vec_top = text_topright[LEFTMOST_COL_ALIGN] - text_topleft[LEFTMOST_COL_ALIGN]
    vec_bottom = text_bottomright[LEFTMOST_COL_ALIGN] - text_bottomleft[LEFTMOST_COL_ALIGN]
    
    up = pt(0, 1)
    right = pt(1, 0)
    
    left_angle = vecangle(vec_left, up)
    right_angle = vecangle(vec_right, up)
    top_angle = vecangle(vec_top, right)
    bottom_angle = vecangle(vec_bottom, right)
    
    angles = np.array((left_angle, right_angle, top_angle, bottom_angle))
    rng = np.max(angles) - np.min(angles)
    
    if rng > MAX_PAGE_ROTATION_RANGE:
        warning('big range of values for page rotation estimation: %f degrees' % math.degrees(rng))
    
    return np.median(angles)


def fix_rotation(p):
    """
    :param p page or subpage
    """
    # TODO: offset beachten für sub_right!
    
    page_corners_texts = texts_at_page_corners(p)
    page_rot = page_rotation_angle(*page_corners_texts)
    
    print("page rotation for page %d is %f" % (p['number'], math.degrees(page_rot)))
    
    text_topleft, text_bottomleft = page_corners_texts[0], page_corners_texts[3]
    bottomline_pts = (
        pt(0, p['height']),
        pt(p['width'], p['height'])
    )
    
    rot_about = pointintersect(text_topleft['topleft'], text_bottomleft['bottomleft'], *bottomline_pts,
                               check_in_segm=False)
                               
    for t in p['texts']:
        t_pt = pt(t['left'], t['top'])
        
        # rotate back
        t_pt_rot = vecrotate(t_pt, -page_rot, rot_about)
        
        # update text dict
        update_text_dict_pos(t, t_pt_rot, update_node=True)


def sorted_by_attr(texts, attr):
    return sorted(texts, key=lambda x: x[attr])
