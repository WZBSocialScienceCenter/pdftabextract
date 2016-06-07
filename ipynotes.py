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

