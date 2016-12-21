# -*- coding: utf-8 -*-
"""
Functions for handling textboxes or making calculations with textboxes' properties.

Created on Wed Dec 21 11:03:14 2016

@author: mkonrad
"""

import numpy as np

from pdftabextract.common import DIRECTION_HORIZONTAL, DIRECTION_VERTICAL


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