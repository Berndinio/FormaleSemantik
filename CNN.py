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

import matplotlib
matplotlib.use('Agg')
from matplotlib.image import imread
from matplotlib import pyplot as plt


def plot_history(train_acc_history, val_acc_history, train_los_history, val_los_history, allModelsInOnePlotPng = True, MultiPlotPng = True, name1='modelsGraph.png', name2='modelsGraph2.png'):
    # Plot training & validation accuracy and loss values
    if allModelsInOnePlotPng:
        legendList = []
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        for i in range (0, len(train_acc_history)):
            ax1.plot(train_acc_history[i])
            ax2.plot(val_acc_history[i])
            ax3.plot(train_los_history[i])
            ax4.plot(val_los_history[i])
            legendList.append('Model ' + str(i+1))
        ax1.set_title('Training Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(legendList, loc='upper left')

        ax2.set_title('Validation Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(legendList, loc='upper left')

        ax3.set_title('Training Loss')
        ax3.set_ylabel('Loss')
        ax3.set_xlabel('Epoch')
        ax3.legend(legendList, loc='upper left')

        ax4.set_title('Validation Loss')
        ax4.set_ylabel('Loss')
        ax4.set_xlabel('Epoch')
        ax4.legend(legendList, loc='upper left')
#       plt.tight_layout(pad=0.8, w_pad=0.7, h_pad=1.5)
#       plt.subplots_adjust(left=0.3, right=1.9, top=1.9, bottom=0.5)
        fig.set_figheight(7.2*2)
        fig.set_figwidth(15)
        fig.savefig(name1)
        plt.close(fig)
        print('saved modelsGraph.png')

    if MultiPlotPng:
        if len(historyList) > 1:
            fig, axes = plt.subplots(len(train_acc_history), 2)
            for i in range (0, len(historyList)):
                axes[i, 0].plot(train_acc_history[i])
                axes[i, 0].plot(val_acc_history[i])
                axes[i, 1].plot(train_los_history[i])
                axes[i, 1].plot(val_los_history[i])
                axes[i, 0].set_title('Model' + str(i+1) + ' accuracy')
                axes[i, 0].set_ylabel('Accuracy')
                axes[i, 0].set_xlabel('Epoch')
                axes[i, 0].legend(['Train', 'Test'], loc='upper left')
                axes[i, 1].set_title('Model' + str(i+1) + ' loss')
                axes[i, 1].set_ylabel('Loss')
                axes[i, 1].set_xlabel('Epoch')
                axes[i, 1].legend(['Train', 'Test'], loc='upper left')
    #        plt.tight_layout(pad=0.8, w_pad=0.7, h_pad=1.5)
    #            plt.subplots_adjust(left=0.3, right=1.9, top=1.9, bottom=0.5)
            fig.set_figheight(7.2*len(historyList))
            fig.set_figwidth(15)
            fig.savefig(name2)
            plt.close(fig)
            print('saved modelsGraph2.png')
    return 0




# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16, (10,numWordsForInput)),
            nn.BatchNorm2d(16),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(10,1)),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(10,1)),
            nn.BatchNorm2d(32),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(8736,40)#10*numWordsForInput*64, 40)
        self.dropout = nn.Dropout(0.45)
        self.fc2 = nn.Linear(40, num_classes)
        self.dropout = nn.Dropout(0.45)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 8736)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out












#main

cnn1 = True
usedDataSet = "dummy"




#get data
proc = DataProcessing(usedDataSet)
train_dataset = proc.samplesSplitted[0]
valid_dataset = proc.samplesSplitted[1]
test_dataset = proc.samplesSplitted[2]


#train_dataset = train_dataset[:, :,0:2]
#valid_dataset = proc.samplesSplitted[1]
#test_dataset = proc.samplesSplitted[2]


#preprocess data
print(train_dataset.shape)
print(valid_dataset.shape)
print(test_dataset.shape)

samplesize_train = list(train_dataset.size())[0]
samplesize_val = list(valid_dataset.size())[0]
samplesize_test = list(test_dataset.size())[0]
print(samplesize_train)

if cnn1:
    temp = torch.zeros(samplesize_train, 300, 3)
    temp[:, : , :2] = train_dataset[:, : , :2]
    temp[:, : , -1] = train_dataset[:, : , -1]
    train_dataset = temp
    train_dataset= train_dataset.to(Variables.device)

    temp2 = torch.zeros(samplesize_val, 300, 3)
    temp2[:, : , :2] = valid_dataset[:, : , :2]
    temp2[:, : , -1] = valid_dataset[:, : , -1]
    valid_dataset = temp2
    valid_dataset= valid_dataset.to(Variables.device)

    temp3 = torch.zeros(samplesize_test, 300, 3)
    temp3[:, : , :2] = test_dataset[:, : , :2]
    temp3[:, : , -1] = test_dataset[:, : , -1]
    test_dataset = temp3
    test_dataset= test_dataset.to(Variables.device)
else:
    temp4 = torch.zeros(samplesize_train, 300, 46)
    temp4[:, : , :2] = train_dataset[:, : , :2]
    temp4[:, : , :2] = train_dataset[:, : , 45:]
    train_dataset = temp4
    train_dataset= train_dataset.to(Variables.device)

    temp5 = torch.zeros(samplesize_val, 300, 46)
    temp5[:, : , :2] = valid_dataset[:, : , :2]
    temp5[:, : , :2] = train_dataset[:, : , 45:]
    valid_dataset = temp5
    valid_dataset= valid_dataset.to(Variables.device)

    temp6 = torch.zeros(samplesize_test, 300, 46)
    temp6[:, : , :2] = test_dataset[:, : , :2]
    temp6[:, : , :2] = train_dataset[:, : , 45:]
    test_dataset = temp6
    test_dataset= test_dataset.to(Variables.device)


print(train_dataset.shape)
print(valid_dataset.shape)
print(test_dataset.shape)

train_dataset = train_dataset[:, None, :, :]
valid_dataset = valid_dataset[:, None, :, :]
test_dataset = test_dataset[:, None, :, :]


# Device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
num_classes = 81
batch_size = 100
learning_rate = 0.0015
if cnn1:
    inputCnn = (batch_size,1,300,3)
    numWordsForInput = 2
else:
    inputCnn = (batch_size,1,300,89)
    numWordsForInput = 2+43


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


model = ConvNet(num_classes).to(device)

# val model
def evaluate():
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images = data[:,:,:,:-1]
            labels = data[:, :,:, -1]

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += (predicted == labels).sum().item()

        print('Val Accuracy of the model: {} %'.format(100 * running_corrects / total))
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    val_acc_history.append(epoch_acc)
    val_los_history.append(epoch_loss)


#test model
def test():
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images = data[:,:,:,:-1]
            labels = data[:, :,:, -1]

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format(100 * running_corrects / total))
    test_loss = running_loss / total
    test_acc = running_corrects / total
    print("test_acc: ")
    print(test_acc)
    print("test_loss: ")
    print(test_loss)
    text_file = open("Cnn_Output.txt", "w")
    text_file.write("test_acc: %s" % test_acc)
    text_file.write('\n')
    text_file.write("test_loss: %s" % test_loss)
    text_file.close()




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
        images = data[:,:,:,:-1]
        labels = data[:, :,:, -1]
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        #print(data.shape)
        #print(labels.shape)
        #images = np.expand_dims(images, axis=1)
        #labels = np.expand_dims(labels, axis=1)
        labels = torch.squeeze(labels)
        labels = labels[:, :81]
        labels = labels.argmax(dim=1)
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
    evaluate()





plot_history(train_acc_history, val_acc_history, train_los_history, val_los_history, name1='netGraph.png', name2='netGraphMultiPlot.png')
test()
# Save the model checkpoint and graphs
torch.save(model.state_dict(), 'model.ckpt')
