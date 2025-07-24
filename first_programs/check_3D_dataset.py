#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#************************************
# Author : Yann Letourneur
# Date : 22 July 2025
# Description : Checking that the 3D dataset is well organized with no missing files nor nomenclature errors
# Usage : python3 check_3D_dataset.py
#************************************

# Libraries
import os

# Constants
from constants import PATH3D, STATUS, CONTRAST_MEDIATORS, VIDEO_EXTENSION

# Functions
def count_number_of_patients(path):
    """Count the number of patients in the 3D dataset.
    
    Note:
        The dataset should follow the structure precised in the dataset_architecture folder.

    Args:
        path (string) : root path of the 3D dataset

    Returns:
        int : number of patients for each status
    """
    dict = {}
    for status in STATUS:
        dict[status] = 0
        new_path = os.path.join(path, status)
        dict[status] += len(os.listdir(new_path))
    return dict


def count_patients_by_contrast_mediator(path):
    """Count the number of patients by contrast mediator in the 3D dataset
    and verify that there is no other file.
    
    Note:
        The dataset should follow the structure precised in the dataset_architecture folder.

    Args:
        path (string) : root path of the 3D dataset

    Returns:
        int : total number of videos by contrast mediator
    """
    dict = {}
    for status in STATUS:
        dict[status] = {}
        for contrast_mediator in CONTRAST_MEDIATORS:
            dict[status][contrast_mediator] = 0

    for status in STATUS:
        new_path = os.path.join(path, status)
        for patient in os.listdir(new_path):
            patient_path = os.path.join(new_path, patient)
            
            for contrast_mediator in CONTRAST_MEDIATORS:
                contrast_mediator_path = os.path.join(patient_path, contrast_mediator)
                if os.path.isdir(contrast_mediator_path):
                    dict[status][contrast_mediator] += 1
            
    return dict
        
def count_videos_by_contrast_mediator(path):
    """Count the number of videos by contrast mediator in the 3D dataset
    and verify that there is no other file.
    
    Note:
        The dataset should follow the structure precised in the dataset_architecture folder.

    Args:
        path (string) : root path of the 3D dataset

    Returns:
        int : total number of videos by contrast mediator
    """
    dict = {}
    for status in STATUS:
        dict[status] = {}
        for contrast_mediator in CONTRAST_MEDIATORS:
            dict[status][contrast_mediator] = 0

    for status in STATUS:
        new_path = os.path.join(path, status)
        for patient in os.listdir(new_path):
            patient_path = os.path.join(new_path, patient)
            
            for contrast_mediator in CONTRAST_MEDIATORS:
                contrast_mediator_path = os.path.join(patient_path, contrast_mediator)
                if os.path.isdir(contrast_mediator_path):
                    for file in os.listdir(contrast_mediator_path):
                        if file.endswith(VIDEO_EXTENSION):
                            dict[status][contrast_mediator] += 1
            
    return dict        

def count_number_of_videos(path):
    """Count the number of videos in the 3D dataset.
    
    Note:
        The dataset should follow the structure precised in the dataset_architecture folder.

    Args:
        path (string) : root path of the 3D dataset

    Returns:
        int : total number of videos
    """
    dict = {}
    for status in STATUS:
        dict[status] = {}
        for contrast_mediator in CONTRAST_MEDIATORS:
            dict[status][contrast_mediator] = 0
            
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(VIDEO_EXTENSION):
                # Extract status and contrast mediator from the path
                parts = root.split(os.sep)
                print(parts)  # Debugging line to see the parts of the path
                if len(parts) >= 3:
                    status = parts[-2]
                    contrast_mediator = parts[-1]
                    print(f"Found video: {file} in {status}/{contrast_mediator}")
                    if status in STATUS and contrast_mediator in CONTRAST_MEDIATORS:
                        dict[status][contrast_mediator] += 1
            
    return dict

def print_dict(dict):
    """Print a dictionary in a readable format."""
    for key, value in dict.items():
        print(f"{key} : {value}")

# Main
if __name__ == "__main__":
    print("Number of patients :", count_number_of_patients(PATH3D), "\nTotal :", sum(count_number_of_patients(PATH3D).values()))
    
    print("\nNumber of patients by contrast mediator :")
    print_dict(count_patients_by_contrast_mediator(PATH3D))
    
    print("\nNumber of videos by contrast mediator :")
    print_dict(count_videos_by_contrast_mediator(PATH3D))
    
    print("\nTotal number of videos :")
    print_dict(count_number_of_videos(PATH3D))
    print("Total :", sum(sum(count_number_of_videos(PATH3D).values(), {}).values()))


