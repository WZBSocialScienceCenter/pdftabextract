# -*- coding: utf-8 -*-
"""
Functions for handling textboxes or making calculations with textboxes' properties.

Created on Wed Dec 21 11:03:14 2016

@author: mkonrad
"""

import numpy as np

from pdftabextract.common import sorted_by_attr, DIRECTION_HORIZONTAL, DIRECTION_VERTICAL


def border_positions_and_spans_from_texts(texts, direction):
    """
    From a list of textboxes in <texts>, get the border positions and text box spans for the respective direction.
    For vertical direction, return the text boxes' top and bottom border positions and the text boxes' heights.
    For horizontal direction, return the text boxes' left and right border positions and the text boxes' widths.
    
    <direction> must be DIRECTION_HORIZONTAL or DIRECTION_VERTICAL from pdftabextract.common
    
    Border positions are returned as sorted NumPy array, spans are returned as (unsorted) NumPy array
    """
    if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
        raise ValueError("direction must be  DIRECTION_HORIZONTAL or DIRECTION_VERTICAL (see pdftabextract.common)")
    
    if direction == DIRECTION_VERTICAL:
        attr_lo = 'top'
        attr_hi = 'bottom'
    else:
        attr_lo = 'left'
        attr_hi = 'right'        
    
    positions = []
    spans = []
    for t in texts:
        val_lo = t[attr_lo]
        val_hi = t[attr_hi]
        positions.extend((val_lo, val_hi))
        spans.append(val_hi - val_lo)
    
    return np.array(sorted(positions)), np.array(spans)


def split_texts_by_positions(texts, positions, direction, alignment='high',
                             discard_empty_sections=True,
                             enrich_with_positions=False):
    """
    Split textboxes in <texts> into sections according to <positions> either horizontally or vertically (depending on
    <direction>.)
    
    <positions> must be sorted from low to high!
    """
    if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
        raise ValueError("direction must be  DIRECTION_HORIZONTAL or DIRECTION_VERTICAL (see pdftabextract.common)")
        
    if alignment not in ('low', 'high'):
        raise ValueError("alignment must be  'low' or 'high'")
    
    if len(positions) == 0:
        raise ValueError("positions must be non-empty sequence")
    
    if direction == DIRECTION_VERTICAL:
        attr = 'bottom' if alignment == 'high' else 'top'
    else:
        attr = 'right' if alignment == 'high' else 'left'
    
    prev_pos = 0
    split_texts = []
    n_added_texts = 0
    for pos in positions:
        texts_in_section = [t for t in texts if prev_pos < t[attr] <= pos]
        
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

    
def join_texts(texts, sorted_by='left', glue=' ', strip=True):
    if sorted_by:
        texts = sorted_by_attr(texts, sorted_by)
    
    s = glue.join([t['value'] for t in texts])
    if strip:
        s = s.strip()
    return s