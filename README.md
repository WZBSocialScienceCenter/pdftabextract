# pdftabextract - A set of tools for data mining (OCR-processed) PDFs

July 2016, Markus Konrad <markus.konrad@wzb.eu> / [Berlin Social Science Center](https://www.wzb.eu/en)

## Introduction

This repository contains a set of tools written in Python 3 with the aim to extract tabular data from (OCR-processed)
PDF files. Before these files can be processed they need to be converted to XML files in
[pdf2xml format](http://www.mobipocket.com/dev/pdf2xml/). This is very simple -- see section below for instructions.

After that you can view the extracted text boxes with the
[pdf2xml-viewer](https://github.com/WZBSocialScienceCenter/pdf2xml-viewer) tool if you like. The pdf2xml format can be
loaded and parsed with functions in the `common` submodule. When the pages are skewed, you will need to straighten them
before you can process them further. This can be done with the `fixrotation` submodule. Afterwards you can extract
tabular data from these files and output the data in CSV or JSON format using the `tabextract` submodule.

## Features

* load and parse files in pdf2xml format (`common` submodule)
* straighten skewed pages (`fixrotation` submodule)
* extract tabular data from pdf2xml files and output the data in CSV or JSON format (`tabextract` submodule)

## Requirements

The requirements are listed in `requirements.txt`. You basically need a scientific Python software stack installed
(for example via [Anaconda](https://www.continuum.io/why-anaconda) or [pip](https://pypi.python.org/pypi)) with
the following libraries:

* numpy
* scipy

**The scripts were only tested with Python 3. They might also work with Python 2.x with minor modifications.**

## Converting PDF files to XML files with pdf2xml format

You need to convert your PDFs using the **poppler-utils**, a package which is part of most Linux distributions
and is also available for OSX via Homebrew or MacPorts. From this package we need the command `pdftohtml` and can create
an XML file in pdf2xml format in the following way using the Terminal:

```
pdftohtml -c -i -hidden -xml input.pdf output.xml
```

The arguments *input.pdf* and *output.xml* are your input PDF file and the created XML file in pdf2xml format
respectively. It is important that you specifiy the *-hidden* parameter when you're dealing with OCR-processed
("sandwich") PDFs. You can furthermore add the parameters *-f n* and *-l n* to set only a range of pages to be
converted.

## Usage and examples

For usage and background information, please read my series of blog posts about
[data mining PDFs](https://datascience.blog.wzb.eu/category/pdfs/).

You should have a look at the examples to see how to use the provided functions and configuration settings. Examples are
provided in the *examples* directory. Remember to set the PYTHONPATH according to where you put the
*pdftabextract* package. You can run an example straight from the root dictionary with
`PYTHONPATH=. python examples/process_ocr_output.py` (note: your Python 3 executable might be named `python3`).

Alternatively, you can use an IDE like [Spyder](https://github.com/spyder-ide/spyder).

See the following images of the example input/output:

Original OCR-processed ("sandwich") PDF
![original OCR-processed PDF](https://datascience.blog.wzb.eu/wp-content/uploads/10/2016/07/ocr-output-pdf.png)

Generated (and skewed) pdf2xml file viewed with [pdf2xml-viewer](https://github.com/WZBSocialScienceCenter/pdf2xml-viewer)
![OCR PDF example in the viewer](https://datascience.blog.wzb.eu/wp-content/uploads/10/2016/07/ocr-pdf-example-screenshot.png)

Straightened file
![Straightened OCR PDF example](https://datascience.blog.wzb.eu/wp-content/uploads/10/2016/07/ocr-pdf-example-output-straightened.png)

Extracted data (CSV file imported to LibreOffice)
![Extracted data (CSV file imported to LibreOffice)](https://datascience.blog.wzb.eu/wp-content/uploads/10/2016/07/ocr-pdf-example-csv-output.png)

## License

Apache License 2.0. See LICENSE file.
