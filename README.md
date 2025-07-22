# Internship Project : AI detection of early cervical cancer

```
Author : Yann Letourneur
Date : April to August 2025
```

## Description

This project is an internship project at the KMITL University, Bangkok, Thailand. The goal of this project is to create a model that can detect early cervical cancer from colposcopy images.

## Project structure

- The [dataset_architecture](dataset_architecture) folder contains an explanation of the architecture of the dataset. 

- The [machine_learning](machine_learning) folder contains the code for the machine learning part of the project. 
It is structured in steps, each step corresponding to a different task.

## Usage

This project is made to be used with the `Colpo Dataset` given by the KMITL University. 

Before running any code, change the path to the dataset in the [private_constants.py](private_constants.py) file :

```python
dataset_location = "[the path of the folder containing the folder 'Colpo Dataset']"
```
You may also need to install the libraries listed below if you haven't done so already.

Then, you can run the code in each step by executing the `machine_learning.py` file in the corresponding step folder.

## Libraries

You need to install the following libraries to run the code.

### Option 1: Install from requirements.txt (recommended)
```bash
pip3 install -r requirements.txt
```

### Option 2: Install individually
```bash
pip3 install opencv-python
pip3 install numpy
pip3 install scikit-learn
pip3 install torch torchvision
pip3 install tqdm
```

**Required packages:**
- `opencv-python`: For image processing
- `numpy`: For numerical computations
- `scikit-learn`: For machine learning utilities and metrics
- `torch` + `torchvision`: PyTorch deep learning framework
- `tqdm`: For progress bars during training