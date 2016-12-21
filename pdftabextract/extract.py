# -*- coding: utf-8 -*-
"""
Functions for:
* construction of page grids from column and line positions
* extracting information from textboxes in pages using the page grids

Created on Wed Dec 21 11:28:32 2016

@author: mkonrad
"""

from pdftabextract.geom import rect, pt


def make_grid_from_positions(colpos, rowpos):
    """
    Create a page grid from list of column positions <colpos> and a list of row positions <rowpos>.
    Both positions lists must be sorted from low to high!
    The returned page grid is a list of rows. Each row in turn contains a "grid cell",
    i.e. a rect (see pdftabextract.geom).
    """
    if len(colpos) == 0:
        raise ValueError("List of column positions is empty.")
    if len(rowpos) == 0:
        raise ValueError("List of row positions is empty.")
    
    row_spans = position_spans(rowpos)
    col_spans = position_spans(colpos)    
    
    # create a grid with rectangles of table cells
    grid = []
    
    for top, bottom in row_spans:
        row = []
        for left, right in col_spans:
            cell = rect(pt(left, top), pt(right, bottom))
            row.append(cell)
        grid.append(row)

    return grid


def position_spans(positions):
    p = positions
    return [(p[i-1], v) for i, v in enumerate(p) if i > 0]