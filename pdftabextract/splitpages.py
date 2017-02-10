# -*- coding: utf-8 -*-
"""
Functions for splitting a page into two. Useful when a double page has been scanned and OCR-processed.

Created on Thu Jan 26 16:02:38 2017

@author: mkonrad
"""

import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
import os

import cv2

from .common import DIRECTION_HORIZONTAL, DIRECTION_VERTICAL, update_text_dict_pos


def split_page_texts(p, split_pos, direction=DIRECTION_VERTICAL):
    """
    Split text boxes on a double page <p> at position <split_pos>, the separator line going in direction <direction>.
    Return a 2D tuple with the split pages A and B:
    ((A text boxes, A width, A height), (B text boxes, B width, B height))
    """
    if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
        raise ValueError("invalid value for 'direction': '%s'" % direction)
    
    if direction == DIRECTION_VERTICAL:
        pos_attr = 'left'
        dim_attr = 'width'
        width_a = split_pos
        width_b = p['width'] - split_pos
        height_a = height_b = p['height']
    else:
        pos_attr = 'top'
        dim_attr = 'height'
        height_a = split_pos
        height_b = p['height'] - split_pos
        width_a = width_b = p['width']
        
    texts_a = []
    texts_b = []
    
    for t in p['texts']:
        t = deepcopy(t)
        t_pos =  t[pos_attr] + t[dim_attr]/2
        if t_pos < split_pos:
            texts_a.append(t)
        else:
            if direction == DIRECTION_VERTICAL:
                new_pos = (t['left'] - split_pos, t['top'])
            else:
                new_pos = (t['left'], t['top'] - split_pos)
            update_text_dict_pos(t, new_pos, update_node=True)
            texts_b.append(t)
    
    return (texts_a, width_a, height_a), (texts_b, width_b, height_b)


def create_split_pages_dict_structure(split_pages, save_to_output_path=None):    
    """
    From a list of split pages <split_pages> with tuples containing
    (base double page, (split pages from split_page_texts), (split page images)) form a new page dict structure and new
    XML element structure.
    Return tuple (new XML element tree object, new XML element root, new pages dict with split pages)
    """
    if save_to_output_path:
        output_dir = os.path.dirname(save_to_output_path)
        output_fname = os.path.basename(save_to_output_path)
        
        if not output_fname.endswith('.xml'):
            raise ValueError("file in save_to_output_path '%s' must end with '.xml'" % save_to_output_path)
            
        dot_idx = output_fname.rindex('.')
        basename = output_fname[:dot_idx]
    else:
        output_dir = None
        basename = None
            
    new_root = ET.Element('pdf2xml', {'producer': 'pdftabextract'})
    
    page_num = 1
    pages = OrderedDict()
    for base_p, texts_pair, image_pair in split_pages:
        base_page_elem = base_p['xmlnode']
        
        for (texts, p_w, p_h), img in zip(texts_pair, image_pair):
            # deep copy text boxes
            new_texts = [deepcopy(t) for t in texts]
            
            # create page dict
            new_p_dict = {
                'number': page_num,
                'width': int(round(p_w)),
                'height': int(round(p_h)),
                'texts': new_texts,
            }
            
            # create <page>
            new_page_elem = ET.Element('page', {
                'number': str(page_num),
                'position': 'absolute',
                'top': '0',
                'left': '0',
                'width': str(new_p_dict['width']),
                'height': str(new_p_dict['height'])
            })
            
            # add <fontspec>s
            new_page_elem.extend(deepcopy(base_page_elem.findall('fontspec')))
            
            # save and add <image>
            if save_to_output_path:
                img_h, img_w = img.shape[:2]
                imgfile = '%s_%d.png' % (basename, page_num)
                cv2.imwrite(os.path.join(output_dir, imgfile), img)
                image_elem = ET.Element('image', {
                    'top': '0',
                    'left': '0',
                    'width': str(int(round(img_w))),
                    'height': str(int(round(img_h))),
                    'src': imgfile,
                })
                new_page_elem.append(image_elem)
            else:
                imgfile = None
            
            # add <text>s
            new_page_elem.extend([t['xmlnode'] for t in new_texts])
            
            # add to root
            new_root.append(new_page_elem)
            
            new_p_dict['image'] = imgfile
            new_p_dict['xmlnode'] = new_page_elem
            pages[page_num] = new_p_dict
                 
            page_num += 1
    
    new_tree = ET.ElementTree(new_root)
    
    if save_to_output_path:
        new_tree.write(save_to_output_path)
    
    return new_tree, new_root, pages

