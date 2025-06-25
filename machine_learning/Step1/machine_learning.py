#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#************************************
# Author : Yann Letourneur
# Date : May 25th 2025
# Description : Machine learning pipeline for the 2D dataset.
# Usage : python3 machine_learning.py
#************************************

###############     LIBRARIES     ###############

import os
import cv2 as cv #OpenCV
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import models, transforms

import random

# Visualisation
from tqdm import tqdm
import csv


###############     CONSTANTS     ###############

from constants import PATH2D, DATA_PARTITION, STATUS, CONTRAST_MEDIATORS, IMAGE_EXTENSION, CSV_FOLDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For reproducibility
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if DEVICE == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005

# Easy run
RUN = {
    "mode": "series",  # "sanity" for sanity check, "classic" for classic evaluation, "series" for a series of runs with different hyperparameters
    "model": "resnet18",  # "simple_cnn", "shallow_mlp", "pooled_mlp", "resnet18", "complex_cnn"
    "criterion": "weighted_CEL", # "CEL" (Cross Entropy Loss) or "weighted_CEL". IGNORED IN SANITY CHECKS (no impact)
    "optimizer": "SGD",  # "SGD" or "Adam" 
    "data_augmentation": True,  # Only available for ResNet18
}


###############     DATA PREPROCESSING     ###############

def load_data(data_path):
    """ Load the data from the selected folder into 2 dictionaries : train_data and dev_data.
    
    Args:
        data_path (string): path to the folder containing the data

    Returns:
        data: dictionary containing the data loaded from the folder
    """
    
    train_data = {}
    dev_data = {}
    
    for status in os.listdir(data_path):
        status_path = os.path.join(data_path, status)
        
        nb_patients_status_train = DATA_PARTITION["Train"][status]
        nb_patients_status_dev = DATA_PARTITION["Dev"][status]
        nb_patients_status = len(os.listdir(status_path)) # Total
        assert nb_patients_status == nb_patients_status_train + nb_patients_status_dev, \
            f"Number of patients for status {status} is not consistent: {nb_patients_status} != {nb_patients_status_train} + {nb_patients_status_dev} \
            \n Check the DATA_PARTITION dictionary in constants.py and the number of patients in the folder {status_path}."
            
        nb_patients_loaded = 0
        
        for patient in os.listdir(status_path):
            patient_path = os.path.join(status_path, patient)
            
            # To go from "Cancer", "HSIL", "LSIL" or "Normal" to "Positive" or "Negative"
            simplified_status = simplify_status(status)
                        
            status_int = STATUS.index(simplified_status)
            
            if nb_patients_loaded < nb_patients_status_train:
                # Add to train_data
                data = train_data
            elif nb_patients_loaded < nb_patients_status_train + nb_patients_status_dev:
                # Add to dev_data
                data = dev_data
            # No else case, as the number of patients is checked above
                
            data[patient] = {"status":status_int}
            
            for image in os.listdir(patient_path):
                image_path = os.path.join(patient_path, image)
                image_data = cv.imread(image_path)
                data[patient].update({image:image_data})
                
            nb_patients_loaded += 1
            
    return train_data, dev_data


def simplify_status(status):
    """ Simplify the status of the patient to "Positive" or "Negative".
    
    Args:
        status (string): status of the patient ("Cancer", "HSIL", "LSIL", "Normal")

    Returns:
        class: "Positive" (for Cancer, HSIL and LSIL) or "Negative" (for Normal) 
    """
    match status:
        case "Cancer" | "HSIL" | "LSIL":
            return "Positive"
        case "Normal":
            return "Negative"
        case _: # If the status is not recognized
            raise ValueError(f"Unknown status: {status}. Expected 'Cancer', 'HSIL', 'LSIL' or 'Normal'.")


def extract_labels(data):
    """ Extracting the labels from the data dictionary.
    
    Args:
        data (dictionary): data loaded with the load_data function

    Returns:
        labels: numpy array of labels
    """
    return np.array([data[patient]["status"] for patient in data])


def extract_images(data, contrast_mediator = "saline"):
    """ Extracting the desired images from the data dictionary.
    
    Args:
        data (dictionary): data loaded with the load_data function

    Returns:
        images: numpy array of images
    """
    if contrast_mediator == "All":
        return [[data[patient][contrast_mediator+IMAGE_EXTENSION] for contrast_mediator in CONTRAST_MEDIATORS] for patient in train_data]
    
    # If a specific contrast mediator is specified, extract only that one
    return np.array([data[patient][contrast_mediator+IMAGE_EXTENSION] for patient in data])


def reshape_and_normalize_images(images):
    """ Reshape and normalize the images to be in the range [0, 1].
    
    Args:
        images (numpy array): images to reshape and normalize

    Returns:
        reshaped_images: reshaped and normalized numpy array of images
    """
    # Reshape the images to 4D tensors (batch_size, channels, height, width)
    reshaped_images = images.reshape(images.shape[0], images.shape[3], images.shape[1], images.shape[2])
    
    # Normalize the data
    reshaped_images = reshaped_images.astype(np.float32) / 255.0
    
    return reshaped_images


def preprocess_train_for_resnet(images):
    """
    Resize and normalize train images for ResNet.
    Args:
        images (torch.Tensor): shape (N, 3, H, W)
    Returns:
        torch.Tensor: shape (N, 3, 224, 224)
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed = [preprocess(img.permute(1,2,0).numpy()) for img in images]
    return torch.stack(processed)


def preprocess_test_for_resnet(images):
    """
    Resize and normalize test images for ResNet.
    Args:
        images (torch.Tensor): shape (N, 3, H, W)
    Returns:
        torch.Tensor: shape (N, 3, 224, 224)
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed = [preprocess(img.permute(1,2,0).numpy()) for img in images]
    return torch.stack(processed)


###############     MACHINE LEARNING     ###############

# Evaluation metrics
def accuracy(outs:list, tars:list) -> float:
    """
    Calculating and returning the accuracy between outputs (`outs`) and targets (`tars`),
    where each one is a list of integers like [1,0,1,2,2,1,0,3], with each integer indicating the target label
    """
    accuracy = np.mean([out==tar for out, tar in zip(outs, tars)])
    return accuracy

def UAR(outs:list, tars:list) -> float:
    """
    Calculating and returning the unweighted average recall between outputs (`outs`) and targets (`tars`),
    where each one is a list of integers like [1,0,1,2,2,1,0,3], with each integer indicating the target label
    """
    uar = sklearn.metrics.recall_score(tars, outs, average='macro')
    return uar

### My own models

# Convolutional Neural Network (CNN) model
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU())

        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.ReLU())

        self.layer2_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU())

        self.layer5_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU())

        self.layer5_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = None  # Will be defined after computing feature size

    def _get_conv_output(self, shape):
        # Create a dummy input to infer the output size
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            x = self.layer1(input)
            x = self.layer2_1(x)
            x = self.layer2_2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5_1(x)
            x = self.layer5_2(x)
            return int(np.prod(x.size()))

    def build_fc(self, input_shape, num_classes):
        conv_output_size = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def create_simple_CNN(input_shape, num_classes):
    """
    Args:
        input_shape (PyTorch array): [channels, height, width]
        num_classes (int): number of classes

    Returns:
        model (SimpleCNN): the CNN model
    """
    model = SimpleCNN(input_shape[0])
    model.build_fc(input_shape, num_classes)
    return model

class ComplexCNN(nn.Module):
    def __init__(self, in_channels):
        super(ComplexCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer2_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = None  # Will be defined after computing feature size

    def _get_conv_output(self, shape):
        # Create a dummy input to infer the output size
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            x = self.layer1(input)
            x = self.layer2_1(x)
            x = self.layer2_2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5_1(x)
            x = self.layer5_2(x)
            return int(np.prod(x.size()))

    def build_fc(self, input_shape, num_classes):
        conv_output_size = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def create_complex_CNN(input_shape, num_classes):
    """ Create a ComplexCNN model with Batch Normalization and Dropout.
    
    Args:
        input_shape (PyTorch array): [channels, height, width]
        num_classes (int): number of classes

    Returns:
        model (ComplexCNN): the CNN model
    """
    model = ComplexCNN(input_shape[0]) 
    model.build_fc(input_shape, num_classes)
    return model

# MLPs
class ShallowMLP(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        flat_size = int(np.prod(input_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
 
def create_shallow_mlp(input_shape, num_classes):
    return ShallowMLP(input_shape, num_classes)

class PooledMLP(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Apply global average pooling to reduce each channel to 1 value
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        flat_size = input_shape[0]  # Number of channels after pooling
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.pool(x)
        return self.net(x)
    
def create_pooled_mlp(input_shape, num_classes):
    return PooledMLP(input_shape, num_classes)

### Pretrained models

def create_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the final layer
    
    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    model = model.to(DEVICE)
    return model
    

###############     TRAINING AND TESTING     ###############

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):

    model.to(device)
    for epoch in range(num_epochs): 
        
        model.train() # passage en mode "train" (stockage des gradients, batchnorms, dropout)
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # forward pass
            optimizer.zero_grad() # met à 0.0 les gradients entre les batchs
            outputs = model(inputs) # applique l'inférence forward des résultats
            
            # compute loss
            # targets = targets.squeeze().long() # mise en forme des résponses du batch
            targets = targets.view(-1).long()
            loss = criterion(outputs, targets) # calcul du loss (cross entropie)
            
            # backward pass + optimization step
            loss.backward() # calcule les gradients d(loss)/d(param)
            optimizer.step() # met à jour les poids (ici avec SGD)

                    
def test_model(model, dataloader, console_output=False):
    """ Test the model on the given dataloader and return predictions and targets.

    Args:
        model (Net): the model
        dataloader (PyTorch DataLoader): train_loader_at_eval (for sanity checks) or test_loader (for classic evaluation)
        console_output (bool, optional): Set to True to print the predictions, targets, accuracy and UAR in the console. Defaults to False.

    Returns:
        (list, list): predictions and targets
    """
    model.to("cpu")
    model.eval() # passage en mode évaluation
    y_score = torch.tensor([])

    # data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad(): # quand on est en inférence pour les test, ne pas garder les gradients
        all_targets = []
        all_preds = []
        for inputs, targets in tqdm(dataloader):
            outputs = model(inputs) # outputsize : Batchsize * nbclass
            outputs = outputs.softmax(dim=-1) # softmax sur nbclass
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
        # Evaluation
        acc = accuracy(all_preds, all_targets)
        uar = UAR(all_preds, all_targets)
        
        if console_output:
            print(f"all_preds : {all_preds}")
            print(f"all_targets:{all_targets}")
            print(f'** accuracy : {acc:.3f}')
            print(f'** UAR : {uar:.3f}')
        
        return all_preds, all_targets


def run_classic_evaluation(model, train_dataset, test_dataset,
                           selected_criterion=RUN["criterion"],selected_optimizer=RUN["optimizer"],
                           learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS,
                           console_output=False, csv_filename="predictions.csv"):
    
    print(f"\n==> Running classic evaluation on the {len(train_dataset)} training samples and {len(test_dataset)} testing samples...")
    print(f"Using model: {RUN['model']}, criterion: {selected_criterion}, optimizer: {selected_optimizer}, learning rate: {learning_rate}, num epochs: {num_epochs}")
    
    # Encapsulate the data in a PyTorch DataLoader
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    
    # Loss function
    if selected_criterion == "CEL":
        criterion = nn.CrossEntropyLoss()
    elif selected_criterion == "weighted_CEL":
        # Compute class weights for balanced loss
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train_data.numpy()), y=labels_train_data.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown criterion: {selected_criterion}. Choose 'CEL' or 'weighted_CEL'.")
    
    # Define the optimizer
    if selected_optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif selected_optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {selected_optimizer}. Choose 'SGD' or 'Adam'.")
    
    # To save predictions to CSV
    all_epoch_preds = []
    targets_for_csv = None
    all_epoch_accuracy = []
    all_epoch_uar = []
    
    for i in range(num_epochs):
        print(f"\nEpoch {i+1}/{num_epochs}...")
        train_model(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=1, device=DEVICE)
        preds, targets = test_model(model=model,dataloader=test_loader, console_output=console_output)
        all_epoch_preds.append(preds)
        if targets_for_csv is None:
            targets_for_csv = targets  # Save targets only once
            
        # Store accuracy and UAR for each epoch
        all_epoch_accuracy.append(accuracy(preds, targets))
        all_epoch_uar.append(UAR(preds, targets))
            
    # Write to CSV
    write_predictions_csv(targets_for_csv, all_epoch_preds, all_epoch_accuracy, all_epoch_uar, csv_filename)
    
    print(f"==> Process done\nResults saved to {csv_filename}")


def run_sanity_check(model, feats_train_data, labels_train_data,
                     selected_optimizer=RUN["optimizer"],
                     learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS,
                     console_output=False, csv_filename="sanity_check_predictions.csv"):
    
    # Select a small, balanced subset of the training data
    samples_per_class = 4
    selected_indices = []
    for class_idx in np.unique(labels_train_data.numpy()):
        idxs = np.where(labels_train_data.numpy() == class_idx)[0][:samples_per_class]
        selected_indices.extend(idxs)
    print(f"Selected indices for sanity check: {selected_indices}")
        
    # Create tiny datasets
    feats_tiny = feats_train_data[selected_indices]
    labels_tiny = labels_train_data[selected_indices]
    tiny_dataset = [[feats_tiny[i], labels_tiny[i]] for i in range(len(feats_tiny))]
    tiny_loader = data.DataLoader(dataset=tiny_dataset, batch_size=len(tiny_dataset), shuffle=False)
    
    # Loss function
    # As the dataset is balanced, adding class weights wouldn't make any difference
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer
    if selected_optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif selected_optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {selected_optimizer}. Choose 'SGD' or 'Adam'.")
    
    # To write predictions to CSV
    all_epoch_preds = []
    targets_for_csv = None
    all_epoch_accuracy = []
    all_epoch_uar = []

    print(f"\n==> Sanity check: Overfitting a tiny, balanced subset of {len(tiny_dataset)} samples...")
    print(f"Using model: {RUN['model']}, criterion: CEL, optimizer: {selected_optimizer}, learning rate: {learning_rate}, num epochs: {num_epochs}")
    
    for i in range(num_epochs): 
        print(f"\nSanity Epoch {i+1}/{num_epochs}...")
        train_model(model=model, dataloader=tiny_loader, criterion=criterion, optimizer=optimizer, num_epochs=1, device=DEVICE)
        preds, targets = test_model(model=model, dataloader=tiny_loader, console_output=console_output) # Evaluate on the same tiny set
        all_epoch_preds.append(preds)
        if targets_for_csv is None:
            targets_for_csv = targets  # Save targets only once
            
        # Store accuracy and UAR for each epoch
        all_epoch_accuracy.append(accuracy(preds, targets))
        all_epoch_uar.append(UAR(preds, targets))
            
    # Write to CSV
    write_predictions_csv(targets_for_csv, all_epoch_preds, all_epoch_accuracy, all_epoch_uar, csv_filename)

    print(f"==> Sanity check done.\nResults saved to {csv_filename}")

       
###############     POSTPROCESSING     ###############

def write_predictions_csv(targets, all_epoch_preds, all_epoch_accuracy, all_epoch_uar, filename):
    """
    Write targets and predictions for each epoch to a CSV file.

    Args:
        targets (numpy array): True labels for the dataset
        all_epoch_preds (list of lists): Predictions for each epoch, where each inner list contains predictions for that epoch
        filename (str): Output CSV file path
    """
    num_epochs = len(all_epoch_preds)
    num_samples = len(targets)
    
    if not os.path.exists(CSV_FOLDER):
        os.makedirs(CSV_FOLDER)
    
    filename = os.path.join(CSV_FOLDER, filename)

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        # Write header
        header = ["target"] + [f"e_{i+1}" for i in range(num_epochs)]
        writer.writerow(header)
        # Write rows
        for idx in range(num_samples):
            row = [targets[idx]] + [all_epoch_preds[epoch][idx] for epoch in range(num_epochs)]
            writer.writerow(row)
        
        # Write accuracy and UAR for each epoch
        accuracy = ["accuracy"] + [f"{acc:.3f}".replace('.', ',') for acc in all_epoch_accuracy]
        writer.writerow(accuracy)
        uar = ["UAR     "] + [f"{uar:.3f}".replace('.', ',') for uar in all_epoch_uar]
        writer.writerow(uar)


###############     MAIN     ###############

if __name__ == "__main__":
    
    # Load datasets
    train_data, dev_data = load_data(PATH2D)
    print(f"Loaded {len(train_data)} patients from {PATH2D} into the Train dataset.")
    print(f"Loaded {len(dev_data)} patients from {PATH2D} into the Dev dataset.")
    n_classes = len(STATUS) 
    
    # Extract images and labels
    feats_train_data = extract_images(train_data)
    labels_train_data = extract_labels(train_data)
    feats_dev_data = extract_images(dev_data)
    labels_dev_data = extract_labels(dev_data)
    
    # Reshape the array and normalize the data
    feats_train_data = reshape_and_normalize_images(feats_train_data)
    feats_dev_data = reshape_and_normalize_images(feats_dev_data)
    
    # Convert to PyTorch tensors
    feats_train_data = torch.from_numpy(feats_train_data)
    labels_train_data = torch.from_numpy(labels_train_data)
    feats_dev_data = torch.from_numpy(feats_dev_data)
    labels_dev_data = torch.from_numpy(labels_dev_data)
    
    # Preprocess for ResNet (resize, normalization)
    if RUN["model"] == "resnet18":
        if RUN["data_augmentation"]:
            # Data augmentation
            feats_train_data = preprocess_train_for_resnet(feats_train_data) 
        else:
            # No data augmentation, just resize and normalize
            feats_train_data = preprocess_test_for_resnet(feats_train_data)
        # Resize and normalize dev data
        feats_dev_data = preprocess_test_for_resnet(feats_dev_data)
    
    data_shape = feats_train_data.shape[1:] # (channels, height, width)
    # The original images are [3, 600, 800], resized to [3, 224, 224] for ResNet18 
    
    # Combine features and labels into a single dataset
    train_dataset = [[feats_train_data[i], labels_train_data[i]] for i in range(len(feats_train_data))]
    dev_dataset = [[feats_dev_data[i], labels_dev_data[i]] for i in range(len(feats_dev_data))]
    
    #####   Create the model   #####
    
    ## My own models
    if RUN["model"] == "simple_cnn":
        model = create_simple_CNN(input_shape=data_shape, num_classes=n_classes)
    elif RUN["model"] == "complex_cnn":
        model = create_complex_CNN(input_shape=data_shape, num_classes=n_classes)
    elif RUN["model"] == "shallow_mlp":
        model = create_shallow_mlp(input_shape=data_shape, num_classes=n_classes)
    elif RUN["model"] == "pooled_mlp":
        model = create_pooled_mlp(input_shape=data_shape, num_classes=n_classes)
    ## Pretrained models
    elif RUN["model"] == "resnet18":
        model = create_resnet18(num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {RUN['model']}. Choose from 'cnn', 'shallow_mlp', 'pooled_mlp', or 'resnet18'.")
    
    #####   Evaluate the model   #####
    
    if RUN["mode"] == "classic": # Classic evaluation on the train and dev datasets
        ## Single run with global parameters
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset)
        
    elif RUN["mode"] == "series":
        ## Series of runs with different learning rates and optimizers
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.0005, csv_filename="Adam1_0.0005.csv", selected_optimizer="Adam")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.001, csv_filename="Adam2_0.001.csv", selected_optimizer="Adam")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.005, csv_filename="Adam3_0.005.csv", selected_optimizer="Adam")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.01, csv_filename="Adam4_0.01.csv", selected_optimizer="Adam")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.05, csv_filename="Adam5_0.05.csv", selected_optimizer="Adam")
        
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.0005, csv_filename="SGD1_0.0005.csv", selected_optimizer="SGD")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.001, csv_filename="SGD2_0.001.csv", selected_optimizer="SGD")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.005, csv_filename="SGD3_0.005.csv", selected_optimizer="SGD")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.01, csv_filename="SGD4_0.01.csv", selected_optimizer="SGD")
        run_classic_evaluation(model=model, train_dataset=train_dataset, test_dataset=dev_dataset, learning_rate=0.05, csv_filename="SGD5_0.05.csv", selected_optimizer="SGD")

    elif RUN["mode"] == "sanity": # Sanity check on the train_dataset using a small subset of samples
        run_sanity_check(model=model, feats_train_data=feats_train_data, labels_train_data=labels_train_data)
        
    else:
        raise ValueError(f"Unknown mode: {RUN['mode']}. Choose 'classic' or 'sanity'.")
