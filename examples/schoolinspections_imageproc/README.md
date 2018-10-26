# Checkboxes and crosses: data mining PDFs with the help of image processing

Author: Markus Konrad <markus.konrad@wzb.eu>

Date: Oct 2018

A small example of how to extract data from a PDF with the help of image processing. See also the [companion blog post](https://datascience.blog.wzb.eu/2018/10/26/checkboxes-and-crosses-data-mining-pdfs-with-the-help-of-image-processing/).

Source of the sample PDF: https://www.berlin.de/sen/bildung/schule/berliner-schulen/schulverzeichnis/

Note: This is a stripped down example to work with the sample PDF. Not all PDFs of this source could be read with this,
some minor adjustments and fallbacks are necessary, for example to handle multi-line items in the table rows.

This script uses [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract) to parse the XML
representation of the PDF file. The companion tool
[pdf2xml-viewer](https://github.com/WZBSocialScienceCenter/pdf2xml-viewer) can be used to investigate the XML
representation of the PDF with its text boxes.

I recommend executing this script cell by cell (denoted with `#%%` marks and possible to execute separately with IDEs
like Spyder or PyCharm) in order to understand it.

## Requirements

See `requirements.txt`.

* OpenCV
* pdftabextract
* matplotlib
* NumPy
