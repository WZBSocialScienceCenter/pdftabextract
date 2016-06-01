# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:11:33 2016

@author: mkonrad
"""

import xml.etree.ElementTree as ET

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
            t_top = int(t.attrib['top'])
            t_left = int(t.attrib['left'])
            t_width = int(t.attrib['width'])
            t_height = int(t.attrib['height'])
    
            text = {
                'top': t_top,
                'left': t_left,
                'bottom': t_top + t_height,
                'right': t_left + t_width,
                'width': t_width,
                'height': t_height,
                'center_x': t_left + t_width / 2,
                'center_y': t_top + t_height / 2,
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
    
    return lefttexts, righttexts
    
    
#%%
pages = parse_pages(root)

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5

page = pages[35]

bodytexts = get_bodytexts(page)
div_texts = divide_texts_horizontally(page, bodytexts)

text = div_texts[0]

