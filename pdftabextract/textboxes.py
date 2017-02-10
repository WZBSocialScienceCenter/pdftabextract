# -*- coding: utf-8 -*-
"""
Functions for handling textboxes or making calculations with textboxes' properties.

Created on Wed Dec 21 11:03:14 2016

@author: mkonrad
"""

import math

import numpy as np

from pdftabextract.common import (update_text_dict_pos, sorted_by_attr,
                                  DIRECTION_HORIZONTAL, DIRECTION_VERTICAL, SKEW_X, SKEW_Y)
from pdftabextract.geom import pt, vecrotate


def border_positions_from_texts(texts, direction, only_attr=None):
    """
    From a list of textboxes in <texts>, get the border positions for the respective direction.
    For vertical direction, return the text boxes' top and bottom border positions.
    For horizontal direction, return the text boxes' left and right border positions.
    
    <direction> must be DIRECTION_HORIZONTAL or DIRECTION_VERTICAL from pdftabextract.common.
    
    optional <only_attr> must be either 'low' (only return 'top' or 'left' borders) or 'high' (only return 'bottom' or
    'right').
    
    Border positions are returned as sorted NumPy array.
    """
    if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
        raise ValueError("direction must be DIRECTION_HORIZONTAL or DIRECTION_VERTICAL (see pdftabextract.common)")
        
    if only_attr is not None and only_attr not in ('low', 'high'):
        raise ValueError("only_attr must be either 'low' or 'high' if not set to None (default)")
    
    if direction == DIRECTION_VERTICAL:
        attr_lo = 'top'
        attr_hi = 'bottom'
    else:
        attr_lo = 'left'
        attr_hi = 'right'        
    
    positions = []
    for t in texts:
        if only_attr is None or only_attr == 'low':
            positions.append(t[attr_lo])
        if only_attr is None or only_attr == 'high':
            positions.append(t[attr_hi])
    
    return np.array(sorted(positions))


def split_texts_by_positions(texts, positions, direction, alignment='high',
                             discard_empty_sections=True,
                             enrich_with_positions=False):
    """
    Split textboxes in <texts> into sections according to <positions> either horizontally or vertically (depending on
    <direction>.)
    
    <alignment> must be one of ('low', 'middle', 'high') and is used to determine the text box border (or center
    for 'middle') to use for checking if this text box is inside of a section
    
    <positions> must be sorted from low to high!
    """
    if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
        raise ValueError("direction must be  DIRECTION_HORIZONTAL or DIRECTION_VERTICAL (see pdftabextract.common)")
        
    if alignment not in ('low', 'middle', 'high'):
        raise ValueError("alignment must be  'low' or 'high'")
    
    if len(positions) == 0:
        raise ValueError("positions must be non-empty sequence")
    
    if alignment != 'middle':
        if direction == DIRECTION_VERTICAL:
            attr = 'bottom' if alignment == 'high' else 'top'
        else:
            attr = 'right' if alignment == 'high' else 'left'
        t_in_section = lambda t, p1, p2: p1 < t[attr] <= p2
    else:
        if direction == DIRECTION_VERTICAL:
            t_in_section = lambda t, p1, p2: p1 < t['top'] + t['height'] / 2 <= p2
        else:
            t_in_section = lambda t, p1, p2: p1 < t['left'] + t['width'] / 2 <= p2
    
    prev_pos = -1
    split_texts = []
    n_added_texts = 0
    for pos in positions:
        texts_in_section = [t for t in texts if t_in_section(t, prev_pos, pos)]
        
        if texts_in_section or not discard_empty_sections:
            if enrich_with_positions:
                to_append = (texts_in_section, (prev_pos, pos))
            else:
                to_append = texts_in_section
            split_texts.append(to_append)
            n_added_texts += len(texts_in_section)
        
        prev_pos = pos
    
    assert n_added_texts == len(texts)
    
    return split_texts


def put_texts_in_lines(texts):
    """
    Sort text boxes <texts> vertically first and split them into lines. Sort each line horizontally (left to right).
    
    Returns list of lists, each representing a line with text boxes. Empty lines contain empty lists.
    """
    if not texts:
        return []
    
    mean_text_height = np.mean([t['bottom'] - t['top'] for t in texts])
    
    sorted_ts = list(sorted(texts, key=lambda x: x['top']))     # sort texts vertically
    # create list of vertical spacings between sorted texts
    text_spacings = [t['top'] - sorted_ts[i - 1]['bottom'] for i, t in enumerate(sorted_ts) if i > 0]
    text_spacings.append(0.0)   # last line
    
    # minimum positive spacing is considered to be the general vertical line spacing
    pos_sp = [v for v in text_spacings if v > 0]
    line_vspace = min(pos_sp) if pos_sp else None
    
    # go through all vertically sorted texts
    lines = []
    cur_line = []
    min_vspace_for_break = -mean_text_height / 2   # texts might overlap vertically. if the overlap is more than half
                                                   # the mean text height, it is considered a line break
    for t, spacing in zip(sorted_ts, text_spacings):
        cur_line.append(t)
        
        if spacing >= min_vspace_for_break:    # this is a line break            
            # add all texts to this line sorted by x-position
            lines.append(list(sorted(cur_line, key=lambda x: x['left'])))
            
            # add some empty line breaks if necessary
            if line_vspace:
                lines.extend([] * int(spacing / line_vspace))
            
            # reset
            cur_line = []            

    assert len(cur_line) == 0    # because last line gets a zero-spacing appended
    assert len(texts) == sum(map(len, lines))     # check if all texts were put into lines
    
    return lines

    
def join_texts(texts, sorted_by='left', glue=' ', strip=True):
    """Join strings in text boxes <texts>, sorting them by <sorted_by> and concatenating them using <glue>."""
    if sorted_by:
        texts = sorted_by_attr(texts, sorted_by)
    
    s = glue.join([t['value'] for t in texts])
    if strip:
        s = s.strip()
    return s


def create_text_from_lines(lines, linebreak='\n', linejoin=' ', strip=True):
    """Create a multi-line text string from text boxes <lines> (generated by put_texts_in_lines)"""
    text = ''
    for l in lines:
        text += join_texts(l, glue=linejoin, strip=strip) + linebreak
    
    if strip:
        text = text.strip()

    return text
    
    
def rotate_textboxes(page, page_rot, about_pt):
    """
    Rotate all text boxes in <page> about a point <about_pt> by <page_rot> radians.
    """
    for t in page['texts']:
        t_pt = pt(t['left'], t['top'])
        
        # rotate back
        t_pt_rot = vecrotate(t_pt, page_rot, about_pt)
        
        # update text dict
        update_text_dict_pos(t, t_pt_rot, update_node=True)


def deskew_textboxes(page, skew_radians, skew_direction, about_pt):
    """
    Deskew all text boxes in <page> about a point <about_pt> by <skew_radians> radians in direction <skew_direction>.
    """
    if skew_direction not in (SKEW_X, SKEW_Y):
        raise ValueError("invalid parameter value '%s' for skew_direction" % skew_direction)
    
    for t in page['texts']:
        if skew_direction == SKEW_X:
            x = t['top'] + t['height'] / 2
            ref_idx = 1
            a = -1
        else:
            x = t['left'] + t['width'] / 2
            ref_idx = 0
            a = 1

        # x, y have nothing to do with the x and y in a cartesian coord. system
        # y is the coordinate that gets changed depending on x
        d = x - about_pt[ref_idx]
        y_diff = a * math.sin(skew_radians) * d
        
        
        if skew_direction == SKEW_X:
            pt_deskewed = pt(t['left'] + y_diff, t['top'])
        else:
            pt_deskewed = pt(t['left'], t['top'] + y_diff)
        
        update_text_dict_pos(t, pt_deskewed, update_node=True)

