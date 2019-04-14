# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 02:40:46 2019

@author: Robin
"""

import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from preprocessing import DataProcessing
from torch.utils.data import Dataset, DataLoader
from Variables import Variables
import torch.optim as optim

#get data
proc = DataProcessing("dummy2")
train_dataset = proc.samplesSplitted[0]
valid_dataset = proc.samplesSplitted[1]
test_dataset = proc.samplesSplitted[2]


#preprocess data
print(train_dataset.shape)
print(valid_dataset.shape)
print(test_dataset.shape)

#np.squeeze(train_dataset)
#np.squeeze(valid_dataset)
#np.squeeze(test_dataset)
#test = train_dataset[:, :,0:2]
#print(test.shape)




# Device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 1
num_classes = 80
batch_size = 200
learning_rate = 0.001
numWordsForInput = 2



# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d((batch_size,1,300,89),16, (10,numWordsForInput)),
            nn.BatchNorm2d(16),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(10,numWordsForInput)),
            nn.BatchNorm2d(32),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)



# Train the model

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_acc_history = []
val_acc_history = []
train_los_history = []
val_los_history = []
#best_model_wts = copy.deepcopy(model.state_dict())
#best_acc = 0.0
    
    
total_step = len(train_loader)
running_loss = 0.0
running_corrects = 0
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        #print(i)
        images = data[:,:,:-1]
        labels = data[:, :, -1]
        
        #print(data.shape)
        #print(images.shape)
        #print(labels.shape)
        #images = np.expand_dims(images, axis=1)
        #labels = np.expand_dims(labels, axis=1)
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_corrects += (predicted == labels).sum().item()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    epoch_loss = running_loss / batch_size
    epoch_acc = running_corrects / batch_size
    train_acc_history.append(epoch_acc)
    train_los_history.append(epoch_loss)
    

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
running_loss = 0.0
running_corrects = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images = data[:,:,:-1]
        labels = data[:, :, -1]
        
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        total += labels.size(0)
        running_loss += loss.item()
        running_corrects += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
epoch_loss = running_loss / total
epoch_acc = running_corrects / total
val_acc_history.append(epoch_acc)
val_los_history.append(epoch_loss)



# Save the model checkpoint and graphs
#torch.save(model.state_dict(), 'model.ckpt')


