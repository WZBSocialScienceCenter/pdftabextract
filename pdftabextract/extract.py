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
from pdftabextract.textboxes import join_texts, create_text_from_lines, put_texts_in_lines


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
    
    row_spans = subsequent_pairs(rowpos)
    col_spans = subsequent_pairs(colpos)    
    
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
            first_cell_rect = row[0]
            row_top = first_cell_rect[0][1]
            row_bottom = first_cell_rect[1][1]
            
            if row_top <= t['top'] <= row_bottom \
                    or row_top <= t['bottom'] <= row_bottom \
                    or (t['top'] <= row_top and t['bottom'] >= row_bottom):
                for j, cell_rect in enumerate(row):
                    c_l, c_t = cell_rect[0]
                    c_r, c_b = cell_rect[1]
                    if c_l <= t['left'] <= c_r or c_l <= t['right'] <= c_r \
                            or (t['left'] <= c_l and t['right'] >= c_r):
                        isect = rectintersect(cell_rect, t_rect, norm_intersect_area='b')
                        assert isect is not None
                        if isect > 0:  # only "touch" is not enough
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


def datatable_to_dataframe(table, split_texts_in_lines=False, **kwargs):
    """
    Create a pandas dataframe using datatable <table> and joining all texts in the individual cells.
    """
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
        col_data = []
        for j in range(n_rows):
            if split_texts_in_lines:
                cell_str = create_text_from_lines(put_texts_in_lines(table[j][i]), **kwargs)
            else:
                cell_str = join_texts(table[j][i], **kwargs)
                
            col_data.append(cell_str)
        
        ser = pd.Series(col_data)
        ser.name = 'col' + str(i + 1).zfill(zfill_n)
        col_series[ser.name] = ser
    
    return pd.DataFrame(col_series)


#%% Helper functions

def subsequent_pairs(l):
    """
    Return subsequent pairs of values in a list <l>, i.e. [(x1, x2), (x2, x3), (x3, x4), .. (xn-1, xn)] for a
    list [x1 .. xn]
    """
    
    return [(l[i-1], v) for i, v in enumerate(l) if i > 0]

