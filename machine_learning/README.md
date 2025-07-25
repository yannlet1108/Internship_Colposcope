### Machine Learning for Cervical Cancer Detection

## Code structure

The code is structured as follows :
- [Step 1](Step1) : **1 input image** -> **2 Classes** : `Positive` (0) and `Negative` (1)
- [Step 2](Step2) : **1 input image** -> **4 Classes** : `Cancer` (0), `HSIL` (1), `LSIL` (2) and `Normal` (3)
- [Step 3](Step3) : **Multiple input images** -> **2 Classes** : `Positive` (0) and `Negative` (1)
- [Step 4](Step4) : **Multiple input images** -> **4 Classes** : `Cancer` (0), `HSIL` (1), `LSIL` (2) and `Normal` (3)
- [Step 5](Step5) : **1 input video** -> **2 Classes** : `Positive` (0) and `Negative` (1)
- [Step 6](Step6) : **1 input video** -> **4 Classes** : `Cancer` (0), `HSIL` (1), `LSIL` (2) and `Normal` (3)
- [Step 7](Step7) : **Multiple input videos** -> **2 Classes** : `Positive` (0) and `Negative` (1)
- [Step 8](Step8) : **Multiple input videos** -> **4 Classes** : `Cancer` (0), `HSIL` (1), `LSIL` (2) and `Normal` (3)

## Dataset

The dataset used in this project is described in the [dataset_architecture](../dataset_architecture) folder.

## How to run the code

You can run the code in each step by executing the `machine_learning.py` file in the corresponding step folder. 

For example, to run Step 1, you can execute the following command:

```bash
cd Step1
python3 machine_learning.py
```

The code will automatically load the dataset from the path specified in the `private_constants.py` file. Make sure to set the correct path to the dataset before running the code.

The results will be saved in the `csv_output` folder in the corresponding step folder (created automatically if necessary). 
