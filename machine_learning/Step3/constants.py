#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#************************************
# Author : Yann Letourneur
# Date : May 15th 2025
# Description : Constants for the project
# Usage : In another python file :
#      from constants import ...
#************************************

import sys
sys.path.append("../..")  # Add path to the private_constants.py file

from private_constants import dataset_location

# Path
PATH = dataset_location + "/Colpo Dataset"

PATH2D = PATH + "/2D"

PATH3D = PATH + "/3D"

# Dataset partitioning between Train and Dev sets
# The numbers represent the number of patients in each category
DATA_PARTITION = {"Train": {"Cancer":9,"HSIL":14,"LSIL":11,"Normal":20},
                  "Dev": {"Cancer":2,"HSIL":3,"LSIL":3,"Normal":5}}

# Status of the patients
STATUS = ("Positive", "Negative")

# Contrast mediators = names of the 3 images for each patient
CONTRAST_MEDIATORS = ("acetic_acid", "lugol", "saline")

# Image extension for 2D images
IMAGE_EXTENSION = ".jpg"

# Folder name for csv output files
CSV_FOLDER = "csv_output"
