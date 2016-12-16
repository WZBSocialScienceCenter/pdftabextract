# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:40:39 2016

@author: mkonrad
"""

import math
from logging import warning, error

import numpy as np

from .geom import pt, vecangle, vecrotate, pointintersect

from .common import read_xml, parse_pages, get_bodytexts, divide_texts_horizontally, update_text_dict_pos, \
                    texts_at_page_corners, SKEW_X, SKEW_Y

#%%

_conf = {}  # global configuration settings

def set_config(c):
    global _conf
    _conf = c


def set_config_option(o, v):
    _conf[o] = v

#%%
def fix_rotation(input_xml, corner_box_cond_fns=None, override_angles=None,
                 use_sides=set(('left', 'top', 'right', 'bottom'))):
    override_angles = override_angles or {}
    tree, root = read_xml(input_xml)
    
    # get pages objects    
    pages = parse_pages(root)
        
    pages_bodytexts = {}
    pages_contentlengths = {}
    subpages = {}
    for p_num, page in pages.items():
        # strip off footer and header
        bodytexts = get_bodytexts(page, _conf.get('header_skip', 0), _conf.get('footer_skip', 0))
        
        if _conf.get('divide', 0) != 0:
            page_subpages = divide_texts_horizontally(page, _conf.get('divide'), bodytexts)
        else:
            page_subpages = (page, )
        
        for sub_p in page_subpages:
            p_id = (sub_p['number'], sub_p['subpage'])
            
            pages_bodytexts[p_id] = sub_p['texts']
            contentlength = sum([len(t['value']) for t in sub_p['texts']])
            pages_contentlengths[p_id] = contentlength
        
            subpages[p_id] = sub_p
    mean_contentlength = sum([length for length in pages_contentlengths.values()]) / len(pages_contentlengths)
        
    # fix rotation
    rotation_results = {}
    for p_id, sub_p in subpages.items():
        manual_rot_angle = override_angles.get(p_id, None)
        if manual_rot_angle is not None:
            manual_rot_angle = math.radians(manual_rot_angle)
        
        contentlength = pages_contentlengths[p_id]
        contentlength_mean_dev_ratio = contentlength / mean_contentlength
        
        # only fix rotation of pages that have a certain minimum amount of content (perc. of mean content length)
        if contentlength_mean_dev_ratio >= _conf.get('min_content_length_from_mean', 0):
            rotation_results[p_id] = fix_rotation_for_page(sub_p, corner_box_cond_fns,
                                                           use_sides=use_sides,
                                                           manual_rot_angle=manual_rot_angle)
        else:
            print("skipping page '%d/%s': not enough content (%f perc. of the mean %f)"
                  % (p_id[0], p_id[1], contentlength_mean_dev_ratio * 100, mean_contentlength))

            rotation_results[p_id] = False, 'not_enough_content'
    
    return tree, root, rotation_results

#%%


def page_rotation_angle(text_topleft, text_topright, text_bottomright, text_bottomleft,
                        use_sides=set(('left', 'top', 'right', 'bottom'))):
    up = pt(0, 1)
    right = pt(1, 0)

    angles_list = []
    lcol_align = _conf.get('leftmost_col_align', 'topleft')    
    rcol_align = _conf.get('rightmost_col_align', 'topleft')    
    
    if 'left' in use_sides and text_bottomleft and text_topleft:  # left side
        vec_left = text_bottomleft[lcol_align] - text_topleft[lcol_align]
        vec_left[0] *= -1   # because our coordinate system is upside down
        a = vecangle(vec_left, up)
        a = -a if vec_left[0] < 0 else a
        angles_list.append(a)
    
    if 'right' in use_sides and text_bottomright and text_topright: # right side
        vec_right = text_bottomright[rcol_align] - text_topright[rcol_align]
        vec_right[0] *= -1  # because our coordinate system is upside down
        a = vecangle(vec_right, up)
        a = -a if vec_right[0] < 0 else a
        angles_list.append(a)
        
    if 'top' in use_sides and text_topright and text_topleft:     # top side
        vec_top = text_topright[rcol_align] - text_topleft[lcol_align]
        a = vecangle(vec_top, right)
        a = -a if vec_top[1] < 0 else a
        angles_list.append(a)

    if 'bottom' in use_sides and text_bottomright and text_bottomleft:  # bottom side
        vec_bottom = text_bottomright[rcol_align] - text_bottomleft[lcol_align]
        a = -a if vec_bottom[1] < 0 else a
        angles_list.append(a)
        
    angles = np.array(angles_list)
    if np.sum(~np.isnan(angles)) < 1:   # we need at least one valid angle
        return np.nan
        
    angles = angles[~np.isnan(angles)]
    rng = np.max(angles) - np.min(angles)
        
    if rng > _conf.get('max_page_rotation_range', math.radians(1.0)):
        warning('big range of values for page rotation estimation: %f degrees' % math.degrees(rng))
        warning([math.degrees(a) for a in angles])
    
    return np.median(angles)


def fix_rotation_for_page(p, corner_box_cond_fns, manual_rot_angle=None,
                          use_sides=set(('left', 'top', 'right', 'bottom'))):
    """
    :param p page or subpage
    """    
    p_name = str(p['number'])
    if 'subpage' in p:
        p_name += '/' + str(p['subpage'])
    

    page_corners_texts = texts_at_page_corners(p, corner_box_cond_fns)
                
    if (sum(t is not None for t in page_corners_texts) < 2):
        error("page %s: not enough valid corner texts found - did not fix rotation" % p_name)
        return False, 'not_enough_corners'
        
#    corners = ('topleft', 'topright', 'bottomright', 'bottomleft')
#    for i, t in enumerate(page_corners_texts):
#        if t:
#            infostr = "at %f, %f with value '%s'" % (t['left'], t['top'], t['value'])
#        else:
#            infostr = "is None"
#        print("page %s: corner %s %s" % (p_name, corners[i], infostr))

    if manual_rot_angle is None:        
        page_rot = page_rotation_angle(*page_corners_texts, use_sides=use_sides)
        print("page %s: rotation %f" % (p_name, math.degrees(page_rot)))
    else:
        page_rot = manual_rot_angle
    
    if np.isnan(page_rot):
        print("page %s: rotation could not be identified" % p_name)
        return False, 'rotation_not_identified'
    
    if abs(page_rot) < _conf.get('min_page_rotation_apply', math.radians(0.25)):
        print("page %s: will not fix marginal rotation %f°" % (p_name, math.degrees(page_rot)))
        return False, 'marginal_rotation'
    
    text_topleft, text_bottomleft = page_corners_texts[0], page_corners_texts[3]
   
    if text_topleft is not None and text_bottomleft is not None:    
        bottomline_pts = (
            pt(0, p['height']),
            pt(p['x_offset'] + p['width'], p['height'])
        )
            
        rot_about = pointintersect(text_topleft['topleft'], text_bottomleft['bottomleft'], *bottomline_pts,
                                   check_in_segm=False)
    else:
        rot_about = None

    if rot_about is None or np.sum(np.isnan(rot_about)) != 0:
        print("page %s: rotation point could not be identified or is undefined" % p_name)
        rot_about = pt(p['x_offset'], p['height'])
    
    print("page %s: rotate about %f, %f" % (p_name, rot_about[0], rot_about[1]))
    
    rotate_textboxes(p, -page_rot, rot_about)
    
    return True, str(-math.degrees(page_rot))

#%%

def rotate_textboxes(page, page_rot, about_pt):
    for t in page['texts']:
        t_pt = pt(t['left'], t['top'])
        
        # rotate back
        t_pt_rot = vecrotate(t_pt, page_rot, about_pt)
        
        # update text dict
        update_text_dict_pos(t, t_pt_rot, update_node=True)


def deskew_textboxes(page, skew_radians, skew_direction, about_pt):
    if skew_direction not in (SKEW_X, SKEW_Y):
        raise ValueError("invalid parameter value '%s' for skew_direction" % skew_direction)
    
    for t in page['texts']:
        if skew_direction == SKEW_X:
            x = t['top'] + t['height'] / 2
            ref_idx = 1
            trigon_fn = math.cos
        else:
            x = t['left'] + t['width'] / 2
            ref_idx = 0
            trigon_fn = math.sin

        # x, y have nothing to do with the x and y in a cartesian coord. system
        # y is the coordinate that gets changed depending on x
        d = x - about_pt[ref_idx]
        y_diff = trigon_fn(skew_radians) * d
        
        
        if skew_direction == SKEW_X:
            pt_deskewed = pt(t['left'] + y_diff, t['top'])
        else:
            pt_deskewed = pt(t['left'], t['top'] + y_diff)
        
        update_text_dict_pos(t, pt_deskewed, update_node=True)
