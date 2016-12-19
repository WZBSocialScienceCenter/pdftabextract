# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:49:35 2016

@author: mkonrad
"""


#%%

import xml.etree.ElementTree as ET
from copy import copy
from collections import OrderedDict
import json

import numpy as np

from .geom import pt, ptdist, rect, rectarea

#%% Constants

ROTATION = 'r'
SKEW_X = 'sx'
SKEW_Y = 'sy'


#%% I/O

def read_xml(fname):
    tree = ET.parse(fname)
    root = tree.getroot()
    
    return tree, root

class JSONEncoderPlus(json.JSONEncoder):
    def default(self, o):
       try:
           iterable = iter(o)
       except TypeError:
           pass
       else:
           return list(iterable)
       # Let the base class default method raise the TypeError
       return super().default(self, o)

def save_page_grids(page_grids, output_file):
    with open(output_file, 'w') as f:
        json.dump(page_grids, f, cls=JSONEncoderPlus)

#%% parsing

def parse_pages(root):
    pages = OrderedDict()
    
    for p in root.findall('page'):
        p_num = int(p.attrib['number'])
        
        p_image = p.findall('image')
        if p_image:
            if len(p_image) != 1:
                raise ValueError("invalid number of image tags on page %d" % p_num)
            imgfile = p_image[0].attrib['src']
        else:
            imgfile = None
        
        page = {
            'number': p_num,
            'image': imgfile,
            'width': int(float(p.attrib['width'])),
            'height': int(float(p.attrib['height'])),
            'texts': [],
            'xmlnode': p,
        }
        
        for t in p.findall('text'):
            tdict = create_text_dict(t)
            trect = rect(tdict['topleft'], tdict['bottomright'])
            
            if rectarea(trect) <= 0:    # seems like there are rectangles with zero area
                continue                # -> skip them
                        
            # join all text elements to one string
            tdict['value'] = ' '.join(t.itertext())
                        
            page['texts'].append(tdict)

        pages[p_num] = page

    return pages


def create_text_dict(t, value=None):
    t_width = int(float(t.attrib['width']))
    t_height = int(float(t.attrib['height']))

    text = {
        'width': t_width,
        'height': t_height,
        'value': value,
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

#%% utility functions for textboxes

def get_bodytexts(page, header_ratio=0.0, footer_ratio=1.0):
    miny = page['height'] * header_ratio
    maxy = page['height'] * (1 - footer_ratio)
    
    #if header_ratio != 0.0:
    #    print('page %d/%s: header cutoff at %f' % (page['number'], page['subpage'], miny))
    #if footer_ratio != 1.0:
    #    print('page %d/%s: footer cutoff at %f' % (page['number'], page['subpage'], maxy))
    
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


def mindist_text(texts, origin, pos_attr, cond_fn=None):
    """
    Get the text that minimizes the distance from its position (defined in pos_attr) to <origin> and satisifies
    the condition function <cond_fn> (if not None).
    """
    texts_by_dist = sorted(texts, key=lambda t: ptdist(origin, t[pos_attr]))
    
    if not cond_fn:
        return texts_by_dist[0]
    else:
        for t in texts_by_dist:
            if cond_fn(t):
                return t
    
    return None


def texts_at_page_corners(p, cond_fns):
    """
    :param p page or subpage
    """
    if cond_fns is None:
        cond_fns = (None, ) * 4
    
    x_offset = p['x_offset']
    
    text_topleft = mindist_text(p['texts'], (x_offset, 0), 'topleft', cond_fns[0])
    text_topright = mindist_text(p['texts'], (x_offset + p['width'], 0), 'topright', cond_fns[1])
    text_bottomright = mindist_text(p['texts'], (x_offset + p['width'], p['height']), 'bottomright', cond_fns[2])
    text_bottomleft = mindist_text(p['texts'], (x_offset, p['height']), 'bottomleft', cond_fns[3])
    
    return text_topleft, text_topright, text_bottomright, text_bottomleft

#%% string functions

def rel_levenshtein(s1, s2):
    """Relative Levenshtein distance taking its upper bound into consideration and return a value in [0, 1]"""
    maxlen = max(len(s1), len(s2))
    if maxlen > 0:
        return levenshtein(s1, s2) / float(maxlen)
    else:
        return 0


def levenshtein(source, target):
    """
    Compute Levenshtein-Distance between strings <source> and <target>.
    Taken from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

#%% Other functions
def fill_array_a_with_values_from_b(a, b, fill_indices):
    """
    Fill array <a> with values specified by <fill_indices> from <b>.
    
    Example:
    fill_array_a_with_values_from_b(np.array(list('136')), np.array(list('abcdef')), [1, 3, 4])
    
    result: ['1' 'b' '3' 'd' 'e' '6']
    """
    if type(a) is not np.ndarray:
        raise ValueError("'a' must be NumPy array")
    if type(b) is not np.ndarray:
        raise ValueError("'b' must be NumPy array")
    
    if len(fill_indices) != len(b) - len(a):
        raise ValueError("Invalid number of indices")
    
    mrg = []  # result array
    j = 0     # index in fill_indices
    k = 0     # index in a
    for i in range(len(b)):
        if j < len(fill_indices) and i == fill_indices[j]:
            mrg.append(b[fill_indices[j]])
            j += 1
        else:
            mrg.append(a[k])
            k += 1
    
    return np.array(mrg)


def mode(arr):
    uniques, counts = np.unique(arr, return_counts=True)
    return uniques[np.argmax(counts)]


def sorted_by_attr(vals, attr, reverse=False):
    return sorted(vals, key=lambda x: x[attr], reverse=reverse)


def list_from_attr(vals, attr, **kwargs):
    if 'default' in kwargs:
        return [v.get(attr, kwargs['default']) for v in vals]
    else:
        return [v[attr] for v in vals]

    
def flatten_list(l):
    return sum(l, [])


def any_of_a_in_b(a, b):
    return any(s in b for s in a)

    
def all_of_a_in_b(a, b):
    return all(s in b for s in a)


def updated_dict_copy(d_orig, d_upd):
    d = d_orig.copy()
    d.update(d_upd)
    return d
