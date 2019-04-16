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
        for i in range (0, numModelsTrained):
            ax1.plot(train_acc_history)
            ax2.plot(val_acc_history)
            ax3.plot(train_los_history)
            ax4.plot(val_los_history)
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
        fig, axes = plt.subplots(numModelsTrained, 2)
        if numModelsTrained == 1:
            for i in range (0, numModelsTrained):
                axes[0].plot(train_acc_history)
                axes[0].plot(val_acc_history)
                axes[1].plot(train_los_history)
                axes[1].plot(val_los_history)
                axes[0].set_title('Model' + str(i+1) + ' accuracy')
                axes[0].set_ylabel('Accuracy')
                axes[0].set_xlabel('Epoch')
                axes[0].legend(['Train', 'Validation'], loc='upper left')
                axes[1].set_title('Model' + str(i+1) + ' loss')
                axes[1].set_ylabel('Loss')
                axes[1].set_xlabel('Epoch')
                axes[1].legend(['Train', 'Validation'], loc='upper left')
        else:
            for i in range (0, numModelsTrained):
                axes[i, 0].plot(train_acc_history)
                axes[i, 0].plot(val_acc_history)
                axes[i, 1].plot(train_los_history)
                axes[i, 1].plot(val_los_history)
                axes[i, 0].set_title('Model' + str(i+1) + ' accuracy')
                axes[i, 0].set_ylabel('Accuracy')
                axes[i, 0].set_xlabel('Epoch')
                axes[i, 0].legend(['Train', 'Validation'], loc='upper left')
                axes[i, 1].set_title('Model' + str(i+1) + ' loss')
                axes[i, 1].set_ylabel('Loss')
                axes[i, 1].set_xlabel('Epoch')
                axes[i, 1].legend(['Train', 'Validation'], loc='upper left')
#        plt.tight_layout(pad=0.8, w_pad=0.7, h_pad=1.5)
#            plt.subplots_adjust(left=0.3, right=1.9, top=1.9, bottom=0.5)
        fig.set_figheight(7.2*numModelsTrained)
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
        if(cnn1):
            self.fc1 = nn.Linear(8736,40)#10*numWordsForInput*64, 40)
        else:
            self.fc1 = nn.Linear(4368*44*2,40)
        self.dropout = nn.Dropout(0.45)
        self.fc2 = nn.Linear(40, num_classes)
        self.dropout = nn.Dropout(0.45)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #print(out.shape)
        if(cnn1):
            out = out.view(-1, 8736)
        else:
            out = out.view(-1, 4368*44*2)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        out = self.softmax(out)
        #print(out.shape)
        return out





def preprocess():
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
        temp4[:, : , 2:46] = train_dataset[:, : , 45:]
        train_dataset = temp4
        train_dataset= train_dataset.to(Variables.device)
    
        temp5 = torch.zeros(samplesize_val, 300, 46)
        temp5[:, : , :2] = valid_dataset[:, : , :2]
        temp5[:, : , 2:46] = valid_dataset[:, : , 45:]
        valid_dataset = temp5
        valid_dataset= valid_dataset.to(Variables.device)
    
        temp6 = torch.zeros(samplesize_test, 300, 46)
        temp6[:, : , :2] = test_dataset[:, : , :2]
        temp6[:, : , 2:46] = test_dataset[:, : , 45:]
        test_dataset = temp6
        test_dataset= test_dataset.to(Variables.device)

    print(train_dataset.shape)
    print(valid_dataset.shape)
    print(test_dataset.shape)
   # temp = test_dataset[0:10, :, 2]
    #print(temp.shape)
    #print(temp)
    
    train_dataset = train_dataset[:, None, :, :]
    valid_dataset = valid_dataset[:, None, :, :]
    test_dataset = test_dataset[:, None, :, :]

    # Device config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    
    return samplesize_train, samplesize_val, samplesize_test, train_loader, val_loader, test_loader


# val model
def evaluate(val_acc_history, val_los_history, criterion):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images = data[:,:,:,:-1]
            labels = data[:, :,:, -1]
            labels = torch.tensor(labels, dtype=torch.long, device=Variables.device)
            labels = torch.squeeze(labels)
            labels = labels[:, :81]
            labels = labels.argmax(dim=1)
            images = images.to(Variables.device)
            labels = labels.to(Variables.device)

            outputs = model(images)
            #print(labels.shape)
            #print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += (predicted == labels).sum().item()

    print('Val Accuracy of the model: {} %'.format(100 * running_corrects / samplesize_val))
    epoch_loss = running_loss / samplesize_val
    epoch_acc = running_corrects / samplesize_val
    val_acc_history.append(epoch_acc)
    val_los_history.append(epoch_loss)
    #print(len(val_acc_history))
    return val_acc_history, val_los_history


#test model
def testModel(modelTyp):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images = data[:,:,:,:-1]
            labels = data[:, :,:, -1]
            labels = torch.tensor(labels, dtype=torch.long, device=Variables.device)
            labels = torch.squeeze(labels)
            labels = labels[:, :81]
            labels = labels.argmax(dim=1)
            images = images.to(Variables.device)
            labels = labels.to(Variables.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += (predicted == labels).sum().item()

        #print('Test Accuracy of the model: {} %'.format(100 * running_corrects / total))
    test_loss = running_loss / samplesize_test
    test_acc = running_corrects / samplesize_test
    print("test_acc: ")
    print(test_acc)
    print("test_loss: ")
    print(test_loss)
    if modelTyp == 1:
        text_file = open("Cnn1_Output.txt", "w")
    else:
        text_file = open("Cnn2_Output.txt", "w")
    text_file.write("test_acc: %s" % test_acc)
    text_file.write('\n')
    text_file.write("test_loss: %s" % test_loss)
    text_file.close()
    

def train():
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
    val_acc_history, val_los_history = evaluate(val_acc_history, val_los_history, criterion)
    train_acc_history.append(np.nan)
    train_los_history.append(np.nan)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        count = 0
        for i, data in enumerate(train_loader, 0):
            #print(i)
            images = data[:,:,:,:-1]
            labels = data[:, :,:, -1]
            labels = torch.tensor(labels, dtype=torch.long, device=Variables.device)
           # print(labels.shape)
            labels = torch.squeeze(labels)
            #print(labels.shape)
            labels = labels[:, :81]
            #print(labels.shape)
           # print(labels)
            labels = labels.argmax(dim=1)
           # print(labels.shape)
            images = images.to(Variables.device)
            labels = labels.to(Variables.device)
            #print(labels.shape)
    
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
    
    
    
            loss = criterion(outputs, labels)
    
            #print(outputs.shape)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            running_corrects += (predicted == labels).sum().item()
            if count == 70:
                print(predicted)
                print(labels)
            count = count + 1
    
        
    
        epoch_loss = running_loss / samplesize_train
        epoch_acc = running_corrects / samplesize_train
        train_acc_history.append(epoch_acc)
        train_los_history.append(epoch_loss)
        print ("Epoch: " , epoch+1)
        print("train_epoch_acc" , epoch_acc)
        print("train_epoch_loss" , epoch_loss)
        val_acc_history, val_los_history = evaluate(val_acc_history, val_los_history, criterion)
        
    return train_acc_history, val_acc_history, train_los_history, val_los_history






#main
cnn1 = False
cnn2 = True
usedDataSet = "dummy"


# Hyper parameters
num_epochs = 100
num_classes = 81
batch_size = 100
learning_rate = 0.00007 #0015



numWordsForInput = 2


if cnn1:
    model = ConvNet(num_classes).to(Variables.device)
    inputCnn = (batch_size,1,300,3)
    numWordsForInput = 2
    samplesize_train, samplesize_val, samplesize_test, train_loader, val_loader, test_loader = preprocess()
    train_acc_history, val_acc_history, train_los_history, val_los_history = train()
    # Save model 1 checkpoint
    torch.save(model.state_dict(), 'model1.ckpt')
    testModel(1)

if cnn2:
    model = ConvNet(num_classes).to(Variables.device)
    inputCnn = (batch_size,1,300,46)
    numWordsForInput = 2+43
    samplesize_train, samplesize_val, samplesize_test, train_loader, val_loader, test_loader = preprocess()
    train_acc_history, val_acc_history, train_los_history, val_los_history = train()
    # Save model 2 checkpoint
    torch.save(model.state_dict(), 'model2.ckpt')
    testModel(2)
numModelsTrained = 1
if cnn1 and cnn2:
    numModelsTrained = 2


plot_history(train_acc_history, val_acc_history, train_los_history, val_los_history, name1='netGraph.png', name2='netGraphMultiPlot.png')

