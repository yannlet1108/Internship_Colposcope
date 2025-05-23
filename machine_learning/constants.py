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
sys.path.append("..")  # Add the parent directory to the path

from private_constants import dataset_location

# Path
PATH = dataset_location + "/Colpo Dataset AI" # Dataset split into Train and Dev

PATH2D = PATH + "/2D"
TRAINPATH = PATH2D + "/Train"
DEVPATH = PATH2D + "/Dev"

PATH3D = PATH + "/3D"

# Status of the patients
STATUS = ("Cancer", "HSIL", "LSIL", "Normal")

# Contrast mediators = names of the 3 images for each patient
CONTRAST_MEDIATORS = ("acetic_acid", "lugol", "saline")

# Image extension for 2D images
IMAGE_EXTENSION = "jpg"

