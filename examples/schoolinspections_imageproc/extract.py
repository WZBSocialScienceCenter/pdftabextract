"""
A small example of how to extract data from a PDF with the help of image processing.

Source of the sample PDF: https://www.berlin.de/sen/bildung/schule/berliner-schulen/schulverzeichnis/

Note: This is stripped down example to work with the sample PDF. Not all PDFs of this source could be read with this,
some minor adjustments and fallbacks are necessary, for example to handle multi-line items in the table rows.

This script uses [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract) to parse the XML
representation of the PDF file. The companion tool
[pdf2xml-viewer](https://github.com/WZBSocialScienceCenter/pdf2xml-viewer) can be used to investigate the XML
representation of the PDF with its text boxes.

I recommend executing this script cell by cell (denoted with "#%%" marks and possible to execute separately with IDEs
like Spyder or PyCharm) in order to understand it.

Author: Markus Konrad <markus.konrad@wzb.eu>
Date: Oct 2018
"""

import os
import sys
import re
from pprint import pprint

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pdftabextract.common import read_xml, parse_pages, sorted_by_attr, list_from_attr
from pdftabextract.textboxes import join_texts


#%% A few helper functions


def printerr(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def run_shell_cmd(cmd):
    print('> running shell command: %s' % cmd)
    ret = os.system(cmd)
    if ret != 0:
        printerr('shell command failed with error code %d' % ret)


#%% 1. separate pages with `pdfseparate` if necessary -- not needed here


#%% 2. optional: convert to plain text file, keep layout

# you will notice that the text file does not contain the information that we need: the scores

pdffile = 'samplepage.pdf'
run_shell_cmd('pdftotext -layout %s' % pdffile)


#%% 3. convert to poppler XML format that can be used with pdftabextract tools and pdf2xml-viewer

run_shell_cmd('pdftohtml -xml -i %s' % pdffile)
xmlfile = pdffile[:pdffile.rindex('.')] + '.xml'


#%% 4. Find out the page dimensions of the PDF page

# RE patterns to find page dimensions
# we will need this in order to obtain an image representation (as PNG) of the PDF with exactly the same dimensions
pttrn_xmlpage_height = re.compile(r'^<page.* height="(\d+)".*>')
pttrn_xmlpage_width = re.compile(r'^<page.* width="(\d+)".*>')

# a string template with an image tag that will be added to the XML file
# this is the "background image" which can then be seen behind the text boxes in pdf2xml-viewer
XML_IMAGE_TAG_TEMPLATE = '<image left="0" top="0" width="{width}" height="{height}" src="{src}" />'

# page dimensions
pwidth = None
pheight = None

basename = os.path.splitext(xmlfile)[0]
pngfile = basename + '.png'    # name of the image representation (as PNG) of the PDF that will be
                               # produced in the next cell

# go through each line in the XML file
with open(xmlfile) as xmlf_hndl:
    xml_lines = []
    for line in xmlf_hndl.readlines():
        xml_lines.append(line)
        line = line.strip()
        if pwidth is None or pheight is None:
            # check if this line contains the page dimensions
            match_w = pttrn_xmlpage_width.search(line)
            match_h = pttrn_xmlpage_height.search(line)

            # if yes, extract them
            if match_w and match_h:
                try:
                    pwidth = int(match_w.group(1))
                    pheight = int(match_h.group(1))

                    # add the line with the image tag for pdf2xml-viewer
                    xml_img_tag = XML_IMAGE_TAG_TEMPLATE.format(width=pwidth, height=pheight, src=pngfile)
                    xml_lines.append(xml_img_tag)
                except ValueError:
                    pass

assert pwidth is not None
assert pheight is not None

# store the XML file including the line with the image tag
with open(xmlfile, 'w') as xmlf_hndl:
    xmlf_hndl.writelines(xml_lines)


#%% 5. Convert the PDF to a PNG file

# the arguments to the command specify:
# -png: use png as file format
# -mono: generate binary image (black/white -- not grayscale!)
# -scale-to-x %d -scale-to-y %d: scale to the page dimensions of the PDF found out in the previous cell
#                                -> this makes sure that text boxes' coordinates match coordinates in the PNG

run_shell_cmd('pdftocairo -png -mono -scale-to-x %d -scale-to-y %d %s %s'
              % (pwidth, pheight, pdffile, basename))
run_shell_cmd('mv %s %s' % ((basename + '-1.png'), pngfile))


# -> at this stage, we would load the XML into pdf2xml-viewer in order to see its text boxes

#%% 6. Load and parse the XML representation of the PDF

xmltree, xmlroot = read_xml(xmlfile)

# parse the pages
pages = parse_pages(xmlroot)
assert len(pages) == 1

# we only have a single page
page = pages[1]
del pages

# `page` is now a dict with information about the page itself (width, height, page number) and a list of
# text boxes (in dict element 'texts')
pprint(page)

#%% 7. Load the PNG image representation of the PDF that was generated before

# load image
imgdata = cv2.imread(pngfile)
imgdata = imgdata[:, :, 0]  # binary image input -> select only one of three channels, they are all the same in case
                            # of a binary image anyway

print(imgdata.shape)

imgh, imgw = imgdata.shape

#%% 8. Find out sections (dark blue headers) in the table that contain the scores

# filter for all text boxes with "Bewertung" and sort them from top to bottom
texts_sections = sorted_by_attr([t for t in page['texts'] if t['value'].replace(' ', '') == 'Bewertung'], 'top')

print(len(texts_sections))

# show an example text box
pprint(texts_sections[0])

#%% 9. Parse the data, section by section

# RE pattern to identify the "A B C D" header
pttrn_grade_header = re.compile(r'^[A-D]$')

# RE pattern to identify the item number in front of each row like "1.3" or "E.2"
pttrn_item = re.compile(r'^[1-9A-Z]{1,2}\.\d{1,2}(\.\d{1,2})?')

i_subplot = 1     # we'll generate a plot of the image data inside the checkboxes -- this is the subplot index
item_grades = []  # this will contain the extracted data -- a tuple with:
                  # section number, item in section number, item number string (like "1.3" or "E.2"), item description,
                  # grade (letter A to D)

# go through each section
for i_sec, t_sec in enumerate(texts_sections):
    # get the next section text box or None if it's the last one
    next_t_sec = texts_sections[i_sec + 1] if i_sec < len(texts_sections) - 1 else None

    # identify the top and bottom coordinates of a section
    sec_start_y = t_sec['top']
    sec_end_y = next_t_sec['top'] if next_t_sec else page['height']

    # get all text boxes inside this section (those that are within the range of y-coordinates)
    sec_texts = [t for t in page['texts'] if sec_start_y <= t['top'] <= sec_end_y]

    # find the score header (A B C D) and its column positions
    grade_header_texts = sorted_by_attr([t for t in sec_texts if pttrn_grade_header.search(t['value'].strip())], 'left')

    assert len(grade_header_texts) == 4

    # find the left and right border positions for the area with the checkboxes
    begin_grade_col = grade_header_texts[0]['left']
    end_grade_col = grade_header_texts[-1]['right']

    # get the positions of the individual grade header boxes A to D
    grade_header_pos = list_from_attr(grade_header_texts, 'left')

    # find the text boxes of the item numbers within this section, sort by y-position
    item_texts = sorted_by_attr([t for t in sec_texts if pttrn_item.search(t['value'].strip())], 'top')

    # find the y-positions of each text item
    items_ypos = list_from_attr(item_texts, 'top')

    # with these positions, find the median row height
    median_row_height = np.median(np.diff(items_ypos))

    # go through each item number (i.e. each row)
    for i_item, t_item in enumerate(item_texts):
        # get the next item number box or None if it's the last one
        next_t_item = item_texts[i_item + 1] if i_item < len(item_texts) - 1 else None

        # find out the item number
        item_num = t_item['value'].strip()

        # find out the item row border positions
        item_y = t_item['top']
        item_y_end = next_t_item['top'] if next_t_item else sec_end_y
        if item_y_end == page['height']:
            item_y_end = item_y + median_row_height

        # find out the item description text boxes
        # we apply several criteria for that:
        # 1. the candidate text box `t` is not the item number text box
        # 2. it is in the same row (with a slight offset of -2)
        # 3. it is right to the item number text box (with a slight offset of -5)
        # 4. it contains text
        # 5. it is left to the grades
        descr_texts = [t for t in sec_texts
                       if t is not t_item
                       and item_y - 2 <= t['top'] < item_y_end - 2
                       and t['left'] > t_item['right'] - 5
                       and t['value'].strip()
                       and t_item['left'] <= t['left'] < begin_grade_col]

        # join the text in the text boxes
        item_descr = join_texts(descr_texts)

        # find empty score boxes which approx. show the position of the boxes that contain the grades in the image
        # we apply several criteria for that:
        # 1. the candidate text box `t` is in the same row (with a slight offset of -2)
        # 2. it is an empty text box
        # 3. it's x coordinate is within the range of the grade columns
        empty_grade_boxes = [t for t in sec_texts
                             if item_y - 2 <= t['top'] < item_y_end - 2
                             and t['value'].strip() == ''
                             and begin_grade_col <= t['left'] <= end_grade_col]

        if len(empty_grade_boxes) == 4:   # there are not always grades given
            # parse the empty text boxes that have the approx. position of the checkboxes
            box_fill_ratios = {}
            # go through the positions of the checkbox rectangles
            for box_left, grade_box, grade_letter in zip(grade_header_pos, empty_grade_boxes, 'ABCD'):
                # define the rectangle to be extracted from the image
                # the offsets were found empirically
                box_left = int(min(box_left - 1, imgw))
                box_right = int(min(grade_box['left'] - 3, imgw))
                box_top = int(min(grade_box['top'] - 1, imgh))
                box_bottom = int(min(grade_box['bottom'] - 2, imgh))

                # extract the checkbox image and calculate the ratio of black pixels in this region
                checkbox_img = imgdata[box_top:box_bottom, box_left:box_right]
                ratio_black = np.sum(checkbox_img == 0) / checkbox_img.size
                box_fill_ratios[grade_letter] = ratio_black

                # create a subplot for this checkbox
                subplot = plt.subplot(17, 4, i_subplot)
                autoAxis = subplot.axis()
                rec = plt.Rectangle((-2, -2),
                                    box_right - box_left + 4,
                                    box_bottom - box_top + 4,
                                    fill=False, lw=1, color='red')
                rec = subplot.add_patch(rec)
                rec.set_clip_on(False)
                plt.axis('off')
                plt.imshow(checkbox_img, 'gray', vmin=0, vmax=255)
                i_subplot += 1

            # the box with the highest ratio of black pixels is supposed to be the filled box
            # -> this is the score
            assert len(box_fill_ratios) == 4
            grade, grade_filled_ratio = sorted(box_fill_ratios.items(), key=lambda kv: kv[1], reverse=True)[0]
        else:
            grade = None

        item_grades.append((i_sec, i_item, item_num, item_descr, grade))

#%% 10. Show the plot with the extracted image data of each checkbox

plt.show()

#%% 11. Print the collected data

pprint(item_grades)


