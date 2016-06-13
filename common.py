# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:49:35 2016

@author: mkonrad
"""


#%%

import xml.etree.ElementTree as ET
from copy import copy

import numpy as np

from geom import pt

#%%

def read_xml(fname):
    tree = ET.parse(fname)
    root = tree.getroot()
    
    return tree, root

def parse_pages(root):
    pages = {}
        
    for p in root.findall('page'):
        p_num = int(p.attrib['number'])
        page = {
            'number': p_num,
            'width': int(float(p.attrib['width'])),
            'height': int(float(p.attrib['height'])),
            'x_offset': 0,
            'subpage': None,
            'texts': [],
            'xmlnode': p,
        }
        
        for t in p.findall('text'):
            if not t.text:  # filter out text elements without content
                continue
            
            page['texts'].append(create_text_dict(t))

        pages[p_num] = page

    return pages


def create_text_dict(t):
    t_width = int(float(t.attrib['width']))
    t_height = int(float(t.attrib['height']))

    text = {
        'width': t_width,
        'height': t_height,
        'value': t.text,  # only for easy identification during debugging. TODO: delete
        'xmlnode': t
    }
    
    update_text_dict_pos(text, pt(int(float(t.attrib['left'])), int(float(t.attrib['top']))))
    
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
        t['xmlnode'].attrib['left'] = str(int(round(pos[0])))
        t['xmlnode'].attrib['top'] = str(int(round(pos[1])))


def get_bodytexts(page, header_ratio=0.0, footer_ratio=1.0):
    miny = page['height'] * header_ratio
    maxy = page['height'] * (1 - footer_ratio)
    
    return list(filter(lambda t: t['top'] >= miny and t['bottom'] <= maxy, page['texts']))    


def divide_texts_horizontally(page, divide_ratio, texts=None):
    """
    Divide a page into two subpages by assigning all texts left of a vertical line specified by
    page['width'] * DIVIDE_RATIO to a "left" subpage and all texts right of it to a "right" subpage.
    The positions of the texts in the subpages will stay unmodified and retain their absolute position
    in relation to the page. However, the right subpage has an "offset_x" attribute to later calculate
    the text positions in relation to the right subpage.
    :param page single page dict as returned from parse_pages()
    """
    assert divide_ratio    
    
    if texts is None:
        texts = page['texts']
    
    divide_x = page['width'] * divide_ratio
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


def sorted_by_attr(texts, attr):
    return sorted(texts, key=lambda x: x[attr])