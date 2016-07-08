# Example usage of pdftabextract:
# - fix rotation of skewed pages
# - extract tabular data
# - output to JSON and CSV files
#
# Tested with Python 3.
# Best viewed and executed with Spyder IDE.
#
# July 2016, Markus Konrad <markus.konrad@wzb.eu>
#

import re

from pdftabextract import fixrotation, tabextract

#%% Fix rotation (straighten) of OCR'ed pages

# Define functions to identify text boxes that mark table corners

# Top left und bottom left text boxes must only have the text "G" inside
def cond_topleft_text(t):
    text = t['value'].strip()
    return re.search(r'^(G|WS)$', text) is not None
cond_bottomleft_text = cond_topleft_text

# Define the functions as tuple from top left to bottom left in CW direction
# (Disable corners on the right side -- we don't need them)
cond_disabled = lambda t: False
corner_box_cond_fns = (cond_topleft_text, cond_disabled, cond_disabled, cond_bottomleft_text)

# Fix the rotation
xmltree, xmlroot, rot_results = fixrotation.fix_rotation('examples/ocr-output.pdf.xml', corner_box_cond_fns)

print('FIXROTATION RESULTS:')
for p_id in sorted(rot_results.keys(), key=lambda x: x[0]):
    print("Page %d/%s: %s" % (p_id[0], p_id[1], rot_results[p_id]))

# Write the straightened output XML (just for debugging reasons -- can be viewed with pdf2xml-viewer)
xmltree.write('examples/ocr-output-straightened.pdf.xml')

#%% Extract tabular data from the pages

# Redefine functions to identify text boxes that mark table corners

def cond_topleft_text(t):
    text = t['value'].strip()
    return re.match(r'^\d+', text) is not None
    
def cond_bottomleft_text(t):
    text = t['value'].strip()
    return re.search(r'^(G|WS)$', text) is not None

# Define the functions as tuple from top left to bottom left in CW direction
# (Disable corners on the right side -- we don't need them)
corner_box_cond_fns = (cond_topleft_text, cond_disabled, cond_disabled, cond_bottomleft_text)

# Set configuration options
extract_conf = {
    'header_skip': 0.1,   # ignore top 10% of the page
    'footer_skip': 0.1,   # ignore bottom 10% of the page
    'divide': 0.5,        # two "real" pages per PDF page -> divide page at 50% (in the middle of th page)
    'corner_box_cond_fns': corner_box_cond_fns,  # functions to identify text boxes that mark table corners
    'possible_ncol_range': range(5, 8),  # range of possible number of columns
    'possible_nrow_range': range(2, 15), # range of possible number of rows
    'max_page_col_offset_thresh': 50,    # maximum value of "column offset", i.e. how far can the columns be "off" from the page borders at maximum
    'find_col_clust_property_weights': (1, 5),   # cluster property weights (importance) for finding columns 1:5 for cluster pos. SDs vs. range of values in clusters
    'find_row_clust_property_weights': (1, 1),   # cluster property weights (importance) for finding rows 1:1 for mean distances betw. clusters range vs. SDs of cluster text heights
    'find_row_clust_min_cluster_text_height_thresh': 25,    # minimum row height for found cluster
    'find_row_clust_max_cluster_text_height_thresh': 70,    # maximum row height for found cluster
    'find_row_clust_mean_dists_range_thresh': 30,           # for mean distances betw. clusters range threshold
}

tabextract.set_config(extract_conf)

# Extract tabular data
tabdata, skipped_pages = tabextract.extract_tabular_data_from_xmlroot(xmlroot)

print('SKIPPED PAGES DURING DATA EXTRACTION:')
for p_id in sorted(skipped_pages, key=lambda x: x[0]):
    print("Skipped page %d/%s" % (p_id[0], p_id[1]))

# Save as JSON and as CSV
tabextract.save_tabular_data_dict_as_json(tabdata, 'examples/ocr-output-tabdata.json')
tabextract.save_tabular_data_dict_as_csv(tabdata, 'examples/ocr-output-tabdata.csv')
