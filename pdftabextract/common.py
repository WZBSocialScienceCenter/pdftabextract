# -*- coding: utf-8 -*-
"""
Common functions used in all modules of pdftabextract.

Created on Tue Jun  7 10:49:35 2016

@author: mkonrad
"""


#%%

import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
import json
import os

import numpy as np
import cv2

from .geom import pt, rect, rectarea

#%% Constants

ROTATION = 'r'
SKEW_X = 'sx'
SKEW_Y = 'sy'

DIRECTION_HORIZONTAL = 'h'
DIRECTION_VERTICAL = 'v'


#%% I/O

def read_xml(fname):
    tree = ET.parse(fname)
    root = tree.getroot()
    
    return tree, root

def create_split_pages_dict_structure(split_pages, save_to_output_path=None):    
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
            # create page dict
            new_p_dict = {
                'number': page_num,
                'width': int(round(p_w)),
                'height': int(round(p_h)),
                'texts': texts,
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
            new_page_elem.extend([deepcopy(t['xmlnode']) for t in texts])
            
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

#%% XML parsing / text box dict handling

def parse_pages(root, load_page_nums=None, require_image=False, only_load_topleft_image=True):
    """
    Parses an XML structure in pdf2xml format to extract the pages with their text boxes.
    <root> is the XML tree root
    <load_page_nums> allows to define a sequence of page numbers that should be loaded (by default, all pages
    will be loaded).
    <only_load_topleft_image> if there's more than one background image per page, use the one with top="0" and left="0"
    position.
    """
    pages = OrderedDict()
    
    for p in root.findall('page'):
        p_num = int(p.attrib['number'])
        
        if load_page_nums is not None and p_num not in load_page_nums:
            continue
        
        p_images = p.findall('image')
        if p_images:            
            if len(p_images) == 1:
                imgfile = p_images[0].attrib['src']
            else:
                if not only_load_topleft_image:
                    raise ValueError("multiple images on page %d but only_load_topleft_image was set to False" % p_num)
                for imgtag in p_images:
                    if int(imgtag.attrib['top']) == 0 and int(imgtag.attrib['left']) == 0:
                        imgfile = imgtag.attrib['src']
                        break
                else:
                    raise ValueError("multiple images on page %d but none of it is in the top left corner" % p_num)
        else:
            if require_image:
                raise ValueError("no image given on page %d but require_image was set to True" % p_num)
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


def split_page_texts(p, split_pos, direction=DIRECTION_VERTICAL):
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


def update_text_xmlnode(t, attr, val, round_float=True):
    if round_float:
        val = int(round(val))
    t['xmlnode'].attrib[attr] = str(val)

    
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
        update_text_xmlnode(t, 'left', pos[0])
        update_text_xmlnode(t, 'top', pos[1])


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
        raise TypeError("'a' must be NumPy array")
    if type(b) is not np.ndarray:
        raise TypeError("'b' must be NumPy array")
    
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
    """Return the mode, i.e. most common value, of NumPy array <arr>"""
    uniques, counts = np.unique(arr, return_counts=True)
    return uniques[np.argmax(counts)]


def sorted_by_attr(vals, attr, reverse=False):
    """Sort sequence <vals> by using attribute/key <attr> for each item in the sequence."""
    return sorted(vals, key=lambda x: x[attr], reverse=reverse)


def list_from_attr(vals, attr, **kwargs):
    """Generate a list with all one the items' attributes' in <vals>. The attribute is specified as <attr>."""
    if 'default' in kwargs:
        return [v.get(attr, kwargs['default']) for v in vals]
    else:
        return [v[attr] for v in vals]

    
def flatten_list(l):
    """Flatten a 2D list"""
    return sum(l, [])


def any_a_in_b(a, b):
    return any(s in b for s in a)

    
def all_a_in_b(a, b):
    return all(s in b for s in a)


def updated_dict_copy(d_orig, d_upd):
    d = d_orig.copy()
    d.update(d_upd)
    return d
