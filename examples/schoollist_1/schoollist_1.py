# -*- coding: utf-8 -*-
"""
Dec. 2016, WZB Berlin Social Science Center - https://wzb.eu

@author: Markus Konrad <markus.konrad@wzb.eu>
"""

import os
import re
from math import radians, degrees

import numpy as np
import pandas as pd
import cv2

#%% Some constants
DATAPATH = 'data/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'schoollist_1.pdf.xml'

N_COL_BORDERS = 7
MIN_COL_WIDTH = 194
