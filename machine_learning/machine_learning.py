#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#************************************
# Author : Yann Letourneur
# Date : May 15th 2025
# Description :  **/!\**  TO DO  **/!\**
# Usage : python3 check_2D_dataset.py
#************************************

# Libraries
import os
import cv2 as cv #OpenCV
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.metrics
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

# Visualisation
from tqdm import tqdm
import csv


# Constants
from constants import TRAINPATH, DEVPATH, CONTRAST_MEDIATORS, IMAGE_EXTENSION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 5
LR = 0.001 # learning rate


# Functions
def load_data(folder_path):
    """  **/!\**  TO DO  **/!\**

    Args:
        folder_path (string): path to the Train or Dev dataset

    Returns:
        data: dictionary containing the data loaded from the folder
    """
    
    data = {}
    
    for status in os.listdir(folder_path):
        status_path = os.path.join(folder_path, status)
        # print(f"Loading data for status: {status}")
        
        for patient in os.listdir(status_path):
            patient_path = os.path.join(status_path, patient)
                        
            data[patient] = {"status":status}
            
            for image in os.listdir(patient_path):
                image_path = os.path.join(patient_path, image)
                
                image_data = cv.imread(image_path)
                data[patient].update({image:image_data})
                
    return data


# Functions for evaluation
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


# CNN
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            # nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            # nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))

        """self.fc = nn.Sequential(
            nn.Linear(1853376, 128),
            # nn.Linear(64 * 148 * 198, 128),
            # nn.Linear(64 * 4 * 4, 128),
            # nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            )"""
        
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
        
    def forward(self, x): # description de l'enchaînement des couches dans l'inférence
        x = self.layer1(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5_1(x)
        x = self.layer5_2(x)
        # print(f"Model shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):

    model.to(device)
    for epoch in range(num_epochs): # plusieurs epochs
        #train_correct = 0
        #train_total = 0
        #test_correct = 0
        #test_total = 0

        model.train() # passage en mode "train" (stockage des gradients, batchnorms, dropout)
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # forward pass
            optimizer.zero_grad() # met à 0.0 les gradients entre les batchs
            outputs = model(inputs) # applique l'inférence forward des résultats
            
            # compute loss
            targets = targets.squeeze().long() # mise en forme des résponses du batch
            loss = criterion(outputs, targets) # calcul du loss (cross entropie)
            
            # backward pass + optimization step
            loss.backward() # calcule les gradients d(loss)/d(param)
            optimizer.step() # met à jour les poids (ici avec SGD)
            


def test(model, split):
    model.to("cpu")
    model.eval() # passage en mode évaluation
    y_score = torch.tensor([])

    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad(): # quand on est en inférence pour les test, ne pas garder les gradients
        all_targets = []
        all_preds = []
        for inputs, targets in tqdm(data_loader):
            outputs = model(inputs) # outputsize : Batchsize * nbclass
            outputs = outputs.softmax(dim=-1) # softmax sur nbclass
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
        print(f"all_preds : {all_preds}")
        print(f"all_targets:{all_targets}")

        acc = accuracy(all_preds, all_targets)
        uar = UAR(all_preds, all_targets)
        print(f'** accuracy sur le {split} : {acc:.3f}')
        print(f'** UAR sur le {split} : {uar:.3f}')
        
        return all_preds, all_targets


if __name__ == "__main__":
    
    # Load Train dataset
    train_data = load_data(TRAINPATH)
    print(f"Loaded {len(train_data)} items from {TRAINPATH}.")
    # feats_train_data = [[train_data[patient][contrast_mediator+"."+IMAGE_EXTENSION] for contrast_mediator in CONTRAST_MEDIATORS] for patient in train_data]
    feats_train_data = [train_data[patient]["lugol"+"."+IMAGE_EXTENSION] for patient in train_data]
    labels_train_data = [train_data[patient]["status"] for patient in train_data]
    
    # Load Dev dataset
    dev_data = load_data(DEVPATH)
    print(f"Loaded {len(dev_data)} items from {DEVPATH}.")
    # feats_dev_data = [[dev_data[patient][contrast_mediator+"."+IMAGE_EXTENSION] for contrast_mediator in CONTRAST_MEDIATORS] for patient in dev_data]
    feats_dev_data = [dev_data[patient]["lugol"+"."+IMAGE_EXTENSION] for patient in dev_data]
    labels_dev_data = [dev_data[patient]["status"] for patient in dev_data]
    
    # Convert to numpy arrays
    feats_train_data = np.array(feats_train_data)
    feats_dev_data = np.array(feats_dev_data)
    labels_train_data = np.array(labels_train_data)
    labels_dev_data = np.array(labels_dev_data)
    # print(f"Feats train data shape: {feats_train_data.shape}")
    # print(f"Feats dev data shape: {feats_dev_data.shape}")
    # print(f"Labels train data shape: {labels_train_data.shape}")
    # print(f"Labels dev data shape: {labels_dev_data.shape}")
    
    # Reshape the data to 4D tensors (batch_size, channels, height, width)
    feats_train_data = feats_train_data.reshape(feats_train_data.shape[0], feats_train_data.shape[3], feats_train_data.shape[1], feats_train_data.shape[2])
    feats_dev_data = feats_dev_data.reshape(feats_dev_data.shape[0], feats_dev_data.shape[3], feats_dev_data.shape[1], feats_dev_data.shape[2])
    
    # Normalize the data
    feats_train_data = feats_train_data.astype(np.float32) / 255.0
    feats_dev_data = feats_dev_data.astype(np.float32) / 255.0
    # print(f"Feats train data shape: {feats_train_data.shape}")
    # print(f"Feats dev data shape: {feats_dev_data.shape}")

    # Convert labels to integers
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(labels_train_data)
    labels_train_data = label_encoder.transform(labels_train_data)
    labels_dev_data = label_encoder.transform(labels_dev_data)
    # print(f"Labels train data shape: {labels_train_data.shape}")
    # print(f"Labels dev data shape: {labels_dev_data.shape}")
    # print(labels_dev_data)
    
    # Convert to PyTorch tensors
    feats_train_data = torch.from_numpy(feats_train_data)
    feats_dev_data = torch.from_numpy(feats_dev_data)
    labels_train_data = torch.from_numpy(labels_train_data)
    labels_dev_data = torch.from_numpy(labels_dev_data)
    

    # Définit les données de train et de test
    train_dataset = [[feats_train_data[i], labels_train_data[i]] for i in range(len(feats_train_data))]
    test_dataset = [[feats_dev_data[i], labels_dev_data[i]] for i in range(len(feats_dev_data))]

    # Encapsule ces données sous une forme de dataloader Pytorch
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    
    # Model parameters
    n_channels = 3 # images couleurs, 3 canaux
    n_classes = 4 # les labels sont des entiers de 0 à 3 (4 classes)
    
    criterion = nn.CrossEntropyLoss()
    
    # === SANITY CHECK: Overfit a tiny, balanced subset ===
    # Select 2 samples per class for the sanity check
    samples_per_class = 2
    selected_indices = []
    for class_idx in np.unique(labels_train_data.numpy()):
        idxs = np.where(labels_train_data.numpy() == class_idx)[0][:samples_per_class]
        selected_indices.extend(idxs)
    # Create tiny datasets
    feats_tiny = feats_train_data[selected_indices]
    labels_tiny = labels_train_data[selected_indices]
    tiny_dataset = [[feats_tiny[i], labels_tiny[i]] for i in range(len(feats_tiny))]
    tiny_loader = data.DataLoader(dataset=tiny_dataset, batch_size=len(tiny_dataset), shuffle=True)

    # Define a new model for the sanity check
    model = Net(in_channels=n_channels, num_classes=n_classes)
    model.build_fc(feats_train_data.shape[1:], n_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # To write predictions to CSV
    all_epoch_preds = []
    targets_for_csv = None

    print("\n==> Sanity check: Overfitting a tiny, balanced subset...")
    for i in range(20):  # Train for more epochs to ensure overfitting
        print(f"\nSanity Epoch {i+1}:")
        train_model(model=model, dataloader=tiny_loader, criterion=criterion, optimizer=optimizer, num_epochs=1, device=DEVICE)
        preds, targets = test(model=model, split='train')  # Evaluate on the same tiny set
        all_epoch_preds.append(preds)
        if targets_for_csv is None:
            targets_for_csv = targets  # Save targets only once
            
    print("==> Sanity check done.\n")
    
    # Write to CSV
    with open("sanity_check_predictions.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["target"] + [f"epoch_{i+1}" for i in range(len(all_epoch_preds))]
        writer.writerow(header)
        # Rows
        for idx in range(len(targets_for_csv)):
            row = [targets_for_csv[idx]] + [epoch_preds[idx] for epoch_preds in all_epoch_preds]
            writer.writerow(row)
    
    """
    # Define the model

    model = Net(in_channels=n_channels, num_classes=n_classes)
    
    # definit la loss function utilisée durant le learning
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    # Compute class weights for the loss function
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_train_data.numpy()), y=labels_train_data.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    
    # definit l'optimizer qui traite la backpropagation pendant le training
    # NOTE : l'optimizer est défini SUR UN RESEAU, car il va utiliser les gradients du réseau
    # NOTE : ceci DOIT être exécuté à chaque fois que le modèle est modifié OU quand on retrain de zéro (from scratch)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # print("==> Training ...")
    # train_model(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)
    
    # print("==> Evaluating ...")
    # test(model=model,split='test')
    
    for i in range(NUM_EPOCHS):
        print("\n Epoque",i+1,":")
        train_model(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=1, device=DEVICE)
        test(model=model,split='test')
        
        
    print("==> Process done.")
    """