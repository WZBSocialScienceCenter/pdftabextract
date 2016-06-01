# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:11:33 2016

@author: mkonrad
"""

import xml.etree.ElementTree as ET

tree = ET.parse('testxmls/1992_93.pdf.xml')
root = tree.getroot()

#%%
def parse_pages(root):
    pages = {}
    
    def sorted_by_attr(texts, attr):
        return sorted(texts, key=lambda x: x[attr])
    
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
                'xmlnode': t
            }
            
            page['texts'].append(text)
            
        page['texts_leftright'] = sorted_by_attr(page['texts'], 'left')
        page['texts_topdown'] = sorted_by_attr(page['texts'], 'top')
        pages[p_num] = page
    
