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
xmltree, xmlroot, rot_results = fixrotation.fix_rotation('examples/ocr-output-example.pdf.xml', corner_box_cond_fns)

print('FIXROTATION RESULTS:')
for p_id in sorted(rot_results.keys(), key=lambda x: x[0]):
    print("Page %d/%s: %s" % (p_id[0], p_id[1], rot_results[p_id]))

# Write the straightened output XML (just for debugging reasons -- can be viewed with pdf2xml-viewer)
xmltree.write('examples/ocr-output-example-straightened.pdf.xml')

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

# Extract tabular data
tabdata, skipped_pages = tabextract.extract_tabular_data_from_xmlroot(xmlroot, corner_box_cond_fns)

print('SKIPPED PAGES DURING DATA EXTRACTION:')
for p_id in sorted(skipped_pages, key=lambda x: x[0]):
    print("Skipped page %d/%s" % (p_id[0], p_id[1]))

# Save as JSON and as CSV
tabextract.save_tabular_data_dict_as_json(tabdata, 'examples/ocr-output-example-tabdata.json')
tabextract.save_tabular_data_dict_as_csv(tabdata, 'examples/ocr-output-example-tabdata.csv')
