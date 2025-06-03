# Dataset Architecture

The dataset this project is based on is on the external SSD named `Colpo1`. It contains the folder `Colpo Dataset` that we will consider the root of the dataset.

The dataset in organied this way :

![Global architecture](png/global_architecture.png)

We will work separately on the 2D and 3D data because they have very different structures. 

## 2D

### Overview

The 2D data is organized in the following way :

![2D data architecture](png/2D_data_architecture.png)

Each patient has a folder named `PXXX` where XXX is the patient number. Inside each folder, 3 files represent 3 images of the same patient taken with different *contrast mediators*. The images are named after them : `acetic_acid.jpg`, `lugol.jpg` and `saline.jpg`. 

![2D patient data](png/2D_patient_data.png)

### Machine Learning Deta Partitioning

In order to train a model, the dataset needs to be split into training and validation sets. The dataset is split by patients, meaning that all images of a patient will be in the same set. This partitioning is chosen in the `DATA_PARTITION` dictionary of the `constants.py` file of each step. This dictionary contains the number of patients per status to put in each set (the first ones are in the training set, the last ones in the validation set).

The chosen partitioning is the following (the closest possible to a 80/20 split) :

```python
DATA_PARTITION = {"Train": {"Cancer":9,"HSIL":14,"LSIL":11, "Normal":20},
                  "Dev": {"Cancer":2,"HSIL":3,"LSIL":3,"Normal":5}}
```

![2D AI partition](png/2D_AI_partition.png)


## 3D 

TO DO
