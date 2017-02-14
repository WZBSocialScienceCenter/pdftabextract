# -*- coding: utf-8 -*-
"""
Common functions used in all modules of pdftabextract.

Created on Tue Jun  7 10:49:35 2016

@author: mkonrad
"""


import xml.etree.ElementTree as ET
from collections import OrderedDict
import json

import numpy as np

from .geom import pt, rect, rectarea

#%% Constants

ROTATION = 'r'
SKEW_X = 'sx'
SKEW_Y = 'sy'

DIRECTION_HORIZONTAL = 'h'
DIRECTION_VERTICAL = 'v'


#%% I/O

def read_xml(fname):
    """
    Read an XML file <fname> which can be later parsed with parse_pages. Uses Python's xml.etree.ElementTree.
    Return a tuple with (XML tree object, tree root element)
    """
    tree = ET.parse(fname)
    root = tree.getroot()
    
    return tree, root


class JSONEncoderPlus(json.JSONEncoder):
    """
    Extended JSONEncoder class to be used in save_page_grids.
    """
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
    """
    Save a dict <page_grids> with page number -> grid structure as JSON to <output_file>.
    The grid can be generated with extract.make_grid_from_positions.
    This file can be displayed with pdf2xml-viewer.
    """
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
    
    Return an OrderedDict with page number -> page dict mapping.
    """
    pages = OrderedDict()
    
    for p in root.findall('page'):  # parse all pages
        p_num = int(p.attrib['number'])
        
        if load_page_nums is not None and p_num not in load_page_nums:
            continue
        
        # find all images of the page
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
        
        # create the page dict structure
        page = {
            'number': p_num,
            'image': imgfile,
            'width': int(float(p.attrib['width'])),
            'height': int(float(p.attrib['height'])),
            'texts': [],
            'xmlnode': p,
        }
        
        # add the text boxes to the page
        for t in p.findall('text'):
            tdict = create_text_dict(t)
            try:
                rect(tdict['topleft'], tdict['bottomright'])
            except ValueError:
                # seems like there are rectangles with zero area
                continue                # -> skip them
                        
            # join all text elements to one string
            tdict['value'] = ' '.join(t.itertext())
                        
            page['texts'].append(tdict)

        pages[p_num] = page

    return pages


def create_text_dict(t, value=None):
    """
    From an XML element <t>, create a text box dict structure and return it.
    """
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
    """
    Set the attribute <attr> to <val> of the XML node connected with text box dict <t>.
    """
    if round_float:
        val = int(round(val))
    t['xmlnode'].attrib[attr] = str(val)

    
def update_text_dict_pos(t, pos, update_node=False):
    """
    Update text box <t>'s position and set it to <pos>, where the first element of <pos> is the x and the second is the
    y coordinate.
    If <update_node> is True, also set the respective attributes in the respective XML node of the text box.
    """
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
    Fill array <a> with values from <b> taking values from indicies specified by <fill_indices>.
    
    Example:
    fill_array_a_with_values_from_b(np.array(list('136')), np.array(list('abcdef')), [1, 3, 4])
    
    indices:      1       3   4       <- in "b"
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
    """Return true if any element *s* of <a> also exists in <b>."""
    return any(s in b for s in a)

    
def all_a_in_b(a, b):
    """Return true if all elements *s* of <a> also exist in <b>."""
    return all(s in b for s in a)


def updated_dict_copy(d_orig, d_upd):
    """
    Create a copy of <d_orig>, update it with <d_upd> and return that updated copy.
    """
    d = d_orig.copy()
    d.update(d_upd)
    
    return d
