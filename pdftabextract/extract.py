# -*- coding: utf-8 -*-
"""
Functions for:
* construction of page grids from column and line positions
* extracting information from textboxes in pages using the page grids

Created on Wed Dec 21 11:28:32 2016

@author: mkonrad
"""

from collections import defaultdict, OrderedDict

from pdftabextract.geom import pt, rect, rect_from_text, rectintersect, rectcenter_dist
from pdftabextract.textboxes import join_texts


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


def fit_texts_into_grid(texts, grid, return_unmatched_texts=False):
    """
    Fit text boxes <texts> into the grid <grid>, always choosing the grid cell with the most intersection for a 
    text box.
    Return a data table with the same dimensions as <grid>, each  cell containing a list of text boxes
    (or an empty list).
    """
    # TODO: speed this function up
    n_rows = len(grid)
    
    if n_rows == 0:
        raise ValueError("invalid grid: grid has no rows")
     
    n_cols = len(grid[0])
        
    if n_cols == 0:
        raise ValueError("invalid grid: grid has no columns")
    
    # iterate through the textblocks of this page
    texts_in_cells = defaultdict(lambda: defaultdict(list))
    unmatched_texts = []
    for t in texts:
        t_rect = rect_from_text(t)   # rectangle of the textbox
        
        # find out the cells with which this textbox rectangle intersects
        cell_isects = []
        for i, row in enumerate(grid):
            for j, cell_rect in enumerate(row):
                c_l, c_t = cell_rect[0]
                c_r, c_b = cell_rect[1]
                if (c_l <= t['left'] <= c_r or c_l <= t['right'] <= c_r) and \
                        (c_t <= t['top'] <= c_b or c_t <= t['bottom'] <= c_b):
                    isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
                    assert isect is not None
                    if isect > 0:  # only "touch" is not enough
                        # TODO: only record the best sell intersections
                        cell_isects.append(((i, j), isect, rectcenter_dist(t_rect, cell_rect)))
        
        if len(cell_isects) > 0:
            # find out the cell with most overlap
            max_isect_val = max([x[1] for x in cell_isects])   # best overlap value
            # could be several "best overlap" cells -> choose the closest
            best_isect = sorted([x for x in cell_isects if x[1] == max_isect_val], key=lambda x: x[2])[0]
            best_i, best_j = best_isect[0]
            
            # add this textblock to the table at the found cell index
            texts_in_cells[best_i][best_j].append(t)
        else:  # no intersection at all
            unmatched_texts.append(t)
    
    # generate a table with the texts
    table = []
    n_texts_in_cells = 0
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            ts = texts_in_cells.get(i, {}).get(j, [])
            row.append(ts)
            n_texts_in_cells += len(ts)
        table.append(row)
    
    assert len(texts) == n_texts_in_cells + len(unmatched_texts)
    
    if return_unmatched_texts:
        return table, unmatched_texts
    else:
        return table


def datatable_to_dataframe(table, **join_texts_kwargs):
    import pandas as pd
    
    n_rows = len(table)
    if n_rows == 0:
        raise ValueError('data table must contain rows')
    
    n_cols = len(table[0])
    if n_cols == 0:
        raise ValueError('data table must contain columns')
    
    col_series = OrderedDict()
    zfill_n = len(str(n_cols + 1))
    for i in range(n_cols):
        col_data = [join_texts(table[j][i], join_texts_kwargs) for j in range(n_rows)]
        ser = pd.Series(col_data)
        ser.name = 'col' + str(i + 1).zfill(zfill_n)
        col_series[ser.name] = ser
    
    return pd.DataFrame(col_series)


#%% Helper functions

def position_spans(positions):
    p = positions
    return [(p[i-1], v) for i, v in enumerate(p) if i > 0]