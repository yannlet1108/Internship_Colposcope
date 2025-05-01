#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#************************************
# Author : Yann Letourneur
# Date : 30 Avril 2025
# Description : Checking that the 2D dataset is well organized with no missing files nor nomenclature errors
# Usage : python3 check_2D_dataset.py
#************************************

# Libraries
import os

# Constants
from constants import PATH2D, STATUS, CONTRAST_MEDIATORS, IMAGE_EXTENSION

# Functions
def count_number_of_patients(path):
    """Count the number of patients in the 2D dataset.
    
    Note:
        The dataset should follow the structure precised in the dataset_architecture folder.

    Args:
        path (string) : root path of the 2D dataset

    Returns:
        int : total number of patients
    """
    sum = 0
    for status in STATUS:
        new_path = os.path.join(path, status)
        sum += len(os.listdir(new_path))
    return sum


def count_images_by_contrast_mediator(path):
    """Count the number of images by contrast mediator in the 2D dataset
    and verify that there is no other file.
    
    Note:
        The dataset should follow the structure precised in the dataset_architecture folder.

    Args:
        path (string) : root path of the 2D dataset

    Returns:
        int : total number of images by contrast mediator
    """
    dict = {}
    for CONTRAST_MEDIATOR in CONTRAST_MEDIATORS:
        dict[CONTRAST_MEDIATOR] = 0
        
    for root, dirs, files in os.walk(path):
        for file in files:
            # every file should be in the form of "contrast_mediator.extension
            filename,extension = file.split(".")
            
            # asserting that each file as the right name and extension
            assert filename in CONTRAST_MEDIATORS, "File " + file + " has name " + filename + " instead of one of the contrast mediators : " + str(CONTRAST_MEDIATORS) + "\nin " + root
            assert extension == IMAGE_EXTENSION, "File " + file + " has extension " + extension + " instead of " + IMAGE_EXTENSION + "\nin " + root
            
            # counting the number of images by contrast mediator
            dict[filename] += 1
            
    return dict
        

# Main
if __name__ == "__main__":
    
    print("Total number of patients : ", count_number_of_patients(PATH2D))
    print(count_images_by_contrast_mediator(PATH2D))
    
