# -*- coding: utf-8 -*-
"""
Common functions used in all modules of pdftabextract.

Created on Tue Jun  7 10:49:35 2016

@author: mkonrad
"""


import xml.etree.ElementTree as ET
from collections import OrderedDict
import json
import struct
import imghdr

import numpy as np

from .geom import pt, rect


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


def parse_pages(root, load_page_nums=None, require_image=False, select_image='first', use_images=None):
    """
    Parses an XML structure in pdf2xml format to extract the pages with their text boxes.
    <root> is the XML tree root
    <load_page_nums> allows to define a sequence of page numbers that should be loaded (by default, all pages
    will be loaded).
    <select_image> if there's more than one background image in a page, select an image according to one of these
                   criteria:
                   'first': simply use the first image
                   'topleft': use the one with top="0" and left="0"
    <use_images> pass a dict with page number -> image string mapping to set which images to use for which page
    position.
    
    Return an OrderedDict with page number -> page dict mapping.
    """
    use_images = use_images or {}
    pages = OrderedDict()
    
    for p in root.findall('page'):  # parse all pages
        p_num = int(p.attrib['number'])
        
        if load_page_nums is not None and p_num not in load_page_nums:
            continue
        
        # find all images of the page
        p_images = p.findall('image')
        if p_images:
            use_image = use_images.get(p_num, None)
            if use_image:
                imgfile = use_image
            else:
                if len(p_images) == 1:
                    imgfile = p_images[0].attrib['src']
                else:
                    if select_image == 'first':
                        imgfile = p_images[0].attrib['src']
                    elif select_image == 'topleft':
                        for imgtag in p_images:
                            if int(imgtag.attrib['top']) == 0 and int(imgtag.attrib['left']) == 0:
                                imgfile = imgtag.attrib['src']
                                break
                        else:
                            raise ValueError("multiple images on page %d but none of it is in the top left corner" % p_num)
                    else:
                        raise ValueError('invalid value for parameter `select_image`: "%s"' % select_image)
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


def set_page_image(p, imgfile_src, imgfile_target=None):
    """
    For a page <p>, set the path to an image <imgfile>.
    Modifies <p> in-place.
    """
    if not imgfile_target:
        imgfile_target = imgfile_src

    p['image'] = imgfile_target

    img_size = get_image_size(imgfile_src)
    if not img_size:
        raise ValueError('could not determine image size of file `%s`' % imgfile_src)

    ET.SubElement(p['xmlnode'], 'image', dict(src=imgfile_target, top='0', left='0',
                                              width=str(img_size[0]), height=str(img_size[1])))


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


def update_text_dict_dim(t, dim, update_node=False):
    """
    Update text box <t>'s dimensions and set its width to dim[0] and height to dim[1].
    If <update_node> is True, also set the respective attributes in the respective XML node of the text box.
    """
    if len(dim) != 2:
        raise ValueError('text box dimensions `dim` must be sequence of length 2')

    t_w, t_h = dim
    t_right = t['left'] + t_w
    t_bottom = t['top'] + t_h

    t.update({
        'width': t_w,
        'height': t_h,
        'bottom': t_bottom,
        'right': t_right,
        'bottomleft': np.array((t['left'], t_bottom)),
        'topright': np.array((t_right, t['top'])),
        'bottomright': np.array((t_right, t_bottom)),
    })

    if update_node:
        update_text_xmlnode(t, 'width', t_w)
        update_text_xmlnode(t, 'height', t_h)


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


def test_jpeg(h, f):
    # SOI APP2 + ICC_PROFILE
    if h[0:4] == '\xff\xd8\xff\xe2' and h[6:17] == b'ICC_PROFILE':
        return 'jpeg'
    # SOI APP14 + Adobe
    if h[0:4] == '\xff\xd8\xff\xee' and h[6:11] == b'Adobe':
        return 'jpeg'
    # SOI DQT
    if h[0:4] == '\xff\xd8\xff\xdb':
        return 'jpeg'
imghdr.tests.append(test_jpeg)


def get_image_size(fname):
    """
    Determine the image type of fhandle and return its size.
    Taken from https://stackoverflow.com/a/39778771
    """
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        what = imghdr.what(None, head)
        if what == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif what == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif what == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  #IGNORE:W0703
                return
        else:
            return

        return width, height


def fill_array_a_with_values_from_b(a, b, fill_indices):
    """
    Fill array <a> with values from <b> taking values from indices specified by <fill_indices>.
    
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
