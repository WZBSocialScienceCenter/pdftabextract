# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:40:39 2016

@author: mkonrad
"""

import xml.etree.ElementTree as ET
import math
import re
from copy import copy
from logging import info, warning, error

import numpy as np

from geom import pt, vecangle, vecdist, vecrotate, pointintersect

#%%

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5
# DIVIDE_RATIO = None

LEFTMOST_COL_ALIGN = 'topleft'     # topleft, topright or center
RIGHTMOST_COL_ALIGN = 'topleft'    # topleft, topright or center

MIN_CONTENTLENGTH_MEAN_DEV_RATIO = 0.2

MIN_PAGE_ROTATION_APPLY = math.radians(0.5)     # do not fix page rotation if rotation is below this value
MAX_PAGE_ROTATION_RANGE = math.radians(1.0)     # issue warning when range is too big

#%%
def cond_topleft_text(t):
    return t['value'].strip() == 'G'
cond_bottomleft_text = cond_topleft_text

#def cond_topright_text(t):
#    m = re.search(r'^\d{2}$', t['value'].strip())
#    w = t['width']
#    h = t['height']
#    return m and abs(15 - w) <= 3 and abs(12 - h) <= 2

def cond_topright_text(t):
    return False
cond_bottomright_text = cond_topright_text



#%%
def main():
    tree, root = read_xml('testxmls/1992_93.pdf.xml')
    
    # get pages objects    
    pages = parse_pages(root)
        
    pages_bodytexts = {}
    pages_contentlengths = {}
    subpages = {}
    for p_num, page in pages.items():
        # strip off footer and header
        bodytexts = get_bodytexts(page)
        
        if DIVIDE_RATIO:
            page_subpages = divide_texts_horizontally(page, bodytexts)
        else:
            page_subpages = (page, )
        
        for sub_p in page_subpages:
            if 'subpage' in sub_p:
                p_id = (sub_p['number'], sub_p['subpage'])
            else:
                p_id = (sub_p['number'], )
            pages_bodytexts[p_id] = sub_p['texts']
            contentlength = sum([len(t['value']) for t in sub_p['texts']])
            pages_contentlengths[p_id] = contentlength
        
            subpages[p_id] = sub_p
    mean_contentlength = sum([length for length in pages_contentlengths.values()]) / len(pages_contentlengths)
        
    # fix rotation
    for sub_p in subpages.values():
        contentlength = pages_contentlengths[p_id]
        contentlength_mean_dev_ratio = contentlength / mean_contentlength
        print(contentlength_mean_dev_ratio)
        if contentlength_mean_dev_ratio >= MIN_CONTENTLENGTH_MEAN_DEV_RATIO:
            fix_rotation(sub_p)
        else:
            p_name = str(sub_p['number'])
            if 'subpage' in sub_p:
                p_name += '/' + sub_p['subpage']
            print("skipping page '%s': not enough content (%f% of the mean %d)"
                  % (p_name, contentlength_mean_dev_ratio * 100, mean_contentlength))
        
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
    """
    Divide a page into two subpages by assigning all texts left of a vertical line specified by
    page['width'] * DIVIDE_RATIO to a "left" subpage and all texts right of it to a "right" subpage.
    The positions of the texts in the subpages will stay unmodified and retain their absolute position
    in relation to the page. However, the right subpage has an "offset_x" attribute to later calculate
    the text positions in relation to the right subpage.
    :param page single page dict as returned from parse_pages()
    """
    assert DIVIDE_RATIO    
    
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


def mindist_text(texts, origin, pos_attr, cond_fn=None):
    """
    Get the text that minimizes the distance from its position (defined in pos_attr) to <origin> and satisifies
    the condition function <cond_fn> (if not None).
    """
    texts_by_dist = sorted(texts, key=lambda t: vecdist(origin, t[pos_attr]))
    
    if not cond_fn:
        return texts_by_dist[0]
    else:
        for t in texts_by_dist:
            if cond_fn(t):
                return t
    
    return None


def texts_at_page_corners(p, x_offset, cond_fns):
    """
    :param p page or subpage
    """
    text_topleft = mindist_text(p['texts'], (x_offset, 0), 'topleft', cond_fns[0])
    text_topright = mindist_text(p['texts'], (x_offset + p['width'], 0), 'topright', cond_fns[1])
    text_bottomright = mindist_text(p['texts'], (x_offset + p['width'], p['height']), 'bottomright', cond_fns[2])
    text_bottomleft = mindist_text(p['texts'], (x_offset, p['height']), 'bottomleft', cond_fns[3])
    
    return text_topleft, text_topright, text_bottomright, text_bottomleft


def page_rotation_angle(text_topleft, text_topright, text_bottomright, text_bottomleft, x_offset=0):
    up = pt(0, 1)
    right = pt(1, 0)

    angles_list = []    
    
    if text_bottomleft and text_topleft:  # left side
        vec_left = text_bottomleft[LEFTMOST_COL_ALIGN] - text_topleft[LEFTMOST_COL_ALIGN]
        angles_list.append(vecangle(vec_left, up))
    
    if text_bottomright and text_topright: # right side
        vec_right = text_bottomright[LEFTMOST_COL_ALIGN] - text_topright[LEFTMOST_COL_ALIGN]
        angles_list.append(vecangle(vec_right, up))
        
    if text_topright and text_topleft:     # top side
        vec_top = text_topright[LEFTMOST_COL_ALIGN] - text_topleft[LEFTMOST_COL_ALIGN]
        angles_list.append(vecangle(vec_top, right))

    if text_bottomright and text_bottomleft:  # bottom side
        vec_bottom = text_bottomright[LEFTMOST_COL_ALIGN] - text_bottomleft[LEFTMOST_COL_ALIGN]
        angles_list.append(vecangle(vec_bottom, right))
    
    angles = np.array(angles_list)
    angles = angles[~np.isnan(angles)]
    rng = np.max(angles) - np.min(angles)
    
    if rng > MAX_PAGE_ROTATION_RANGE:
        warning('big range of values for page rotation estimation: %f degrees' % math.degrees(rng))
        warning([math.degrees(a) for a in angles])
    
    return np.median(angles)


def fix_rotation(p):
    """
    :param p page or subpage
    """
    x_offset = p['x_offset'] if 'x_offset' in p else 0
    
    cond_fns = (cond_topleft_text, cond_topright_text, cond_bottomright_text, cond_bottomleft_text)
    page_corners_texts = texts_at_page_corners(p, x_offset, cond_fns)
    
    p_name = str(p['number'])
    if 'subpage' in p:
        p_name += '/' + p['subpage']    

    
    if (sum(t is not None for t in page_corners_texts) < 2):
        error("page %s: not enough valid corner texts found - did not fix rotation" % p_name)
        return False
    
    corners = ('topleft', 'topright', 'bottomright', 'bottomleft')
    for i, t in enumerate(page_corners_texts):
        if t:
            infostr = "at %f, %f with value '%s'" % (t['left'], t['top'], t['value'])
        else:
            infostr = "is None"
        print("page %s: corner %s %s" % (p_name, corners[i], infostr))
    
    page_rot = page_rotation_angle(*page_corners_texts)
    print("page %s: rotation %f" % (p_name, math.degrees(page_rot)))
    
    if page_rot < MIN_PAGE_ROTATION_APPLY:
        print("page %s: will not fix marginal rotation" % p_name)
        return True
    
    text_topleft, text_bottomleft = page_corners_texts[0], page_corners_texts[3]
    bottomline_pts = (
        pt(0, p['height']),
        pt(x_offset + p['width'], p['height'])
    )
    
    rot_about = pointintersect(text_topleft['topleft'], text_bottomleft['bottomleft'], *bottomline_pts,
                               check_in_segm=False)
    
    print("page %s: rotate about %f, %f" % (p_name, rot_about[0], rot_about[1]))
    
    for t in p['texts']:
        t_pt = pt(t['left'], t['top'])
        
        # rotate back
        t_pt_rot = vecrotate(t_pt, -page_rot, rot_about)
        
        # update text dict
        update_text_dict_pos(t, t_pt_rot, update_node=True)
    
    return True


def sorted_by_attr(texts, attr):
    return sorted(texts, key=lambda x: x[attr])
