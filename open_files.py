#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#************************************
# Auteur : Yann Letourneur
# Date : 29 Avril 2025
# Description : 

# Usage : python3 open_files.py
#************************************

# Librairies
import os

# Constantes
PATH = "/media/yann/Colpo1/Colpo Dataset"
PATH2D = PATH + "/2D"
PATH3D = PATH + "/3D"

STATUS = ["Cancer", "HSIL", "LSIL", "Normal"]

CONTRAST_MEDIATORS = ["acetic_acid", "lugol", "saline"]

IMAGE_EXTENSION = ".jpg"

# Functions
def total_number_of_patients(path):
    sum = 0
    for status in STATUS:
        new_path = os.path.join(path, status)
        sum += len(os.listdir(new_path))
    return sum


def count_images(path):
    dict = {}
    for CONTRAST_MEDIATOR in CONTRAST_MEDIATORS:
        dict[CONTRAST_MEDIATOR] = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            filename=file.split(".")[0]
            if filename in CONTRAST_MEDIATORS:
                dict[filename] += 1
            else:
                print("Unknown file name : ", filename)
            
    return dict
        

# Main
if __name__ == "__main__":
    # for status in STATUS:
    #     path = os.path.join(PATH2D, status)
    #     print("***\nStatus :", status, "\nNumber of patients : ", len(os.listdir(path)),"\n***")
    #     for patient_number in os.listdir(path):
    #         print(patient_number)
    #         for file in os.listdir(os.path.join(path, patient_number)):
    #             print(file)
    #             pass
    
    print("Total number of patients : ", total_number_of_patients(PATH2D))
    
    print(count_images(PATH2D))
