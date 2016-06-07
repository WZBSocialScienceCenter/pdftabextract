# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:11:33 2016

@author: mkonrad
"""

import fixrotation


#%%

def cond_topleft_text(t):
    return t['value'].strip() == 'G'
cond_bottomleft_text = cond_topleft_text

#def cond_topright_text(t):
#    m = re.search(r'^\d{2}$', t['value'].strip())
#    w = t['width']
#    h = t['height']
#    return m and abs(15 - w) <= 3 and abs(12 - h) <= 2

def cond_topright_text(t):
    return False
cond_bottomright_text = cond_topright_text

corner_box_cond_fns = (cond_topleft_text, cond_topright_text, cond_bottomright_text, cond_bottomleft_text)
fixrotation.fix_rotation('testxmls/1992_93.pdf.xml', 'testxmls/1992_93_rotback.pdf.xml', corner_box_cond_fns)

#%%

from common import read_xml, parse_pages, get_bodytexts, divide_texts_horizontally

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1
DIVIDE_RATIO = 0.5
# DIVIDE_RATIO = None

LEFTMOST_COL_ALIGN = 'topleft'     # topleft, topright or center
RIGHTMOST_COL_ALIGN = 'topleft'    # topleft, topright or center

MIN_CONTENTLENGTH_MEAN_DEV_RATIO = 0.2

tree, root = read_xml('testxmls/1992_93_rotback.pdf.xml')

# get pages objects    
pages = parse_pages(root)
    
pages_bodytexts = {}
pages_contentlengths = {}
subpages = {}
for p_num, page in pages.items():
    # strip off footer and header
    bodytexts = get_bodytexts(page, HEADER_RATIO, FOOTER_RATIO)
    
    if DIVIDE_RATIO:
        page_subpages = divide_texts_horizontally(page, DIVIDE_RATIO, bodytexts)
    else:
        page_subpages = (page, )
    
    for sub_p in page_subpages:
        if 'subpage' in sub_p:
            p_id = (sub_p['number'], sub_p['subpage'])
        else:
            p_id = (sub_p['number'], )
        pages_bodytexts[p_id] = sub_p['texts']
        contentlength = sum([len(t['value']) for t in sub_p['texts']])
        pages_contentlengths[p_id] = contentlength
    
        subpages[p_id] = sub_p
mean_contentlength = sum([length for length in pages_contentlengths.values()]) / len(pages_contentlengths)

#%%
import matplotlib.pyplot as plt

subpage = subpages[(17, 'left')]

xs = []
ys = []

for t in subpage['texts']:
    xs.append(t['left'])
    ys.append(t['top'])

plt.hist(xs, 100)
plt.hist(ys, 100)