from preprocessing import DataProcessing
from DatasetClasses.RNNDataset import RNNDataset
from models.RNNModel import AttentionLSTM
from torch.utils.data import Dataset, DataLoader
from Variables import Variables
import torch

import torch.nn as nn
import torch.optim as optim
import sys



if __name__ == '__main__':
    batch_size = 10
    num_epochs = 10
    proc = DataProcessing("dummy", loadw2v=False)
    train_dataset = RNNDataset(proc.samplesSplitted[0])
    valid_dataset = RNNDataset(proc.samplesSplitted[1])
    test_dataset = RNNDataset(proc.samplesSplitted[2])
    net = AttentionLSTM().to(Variables.device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # create your optimizer
    optimizer = optim.Adadelta(net.parameters(), lr=1.0, weight_decay=1.0/100000.0)
    criterion = nn.NLLLoss()
    datasetLength = len(train_dataloader)
    datasetLengthValid = len(valid_dataset)
    for epoch in range(num_epochs):
        for i_batch, (inp_x, inp_y) in enumerate(train_dataloader):
            inp_x = inp_x.to(Variables.device)
            inp_y = inp_y.to(Variables.device)
            #compute progress
            progress = int(i_batch/datasetLength * 100)
            sys.stdout.write("\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", Progress: "+'{:>3}'.format(str(progress))+"%")
            sys.stdout.flush()

            optimizer.zero_grad()   # zero the gradient buffers
            output = net(inp_x)
            loss = criterion(output, inp_y)
            loss.backward()
            optimizer.step()    # Does the update
        sys.stdout.write("\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", Progress:  100%")
        print(" ")

        #compute valid accuracy
        accurate = 0
        for i_batch, (inp_x, inp_y) in enumerate(valid_dataloader):
            inp_x = inp_x.to(Variables.device)
            inp_y = inp_y.to(Variables.device)
            output = net(inp_x)
            indices = output.argmax(dim=1)
            accurate += torch.sum(indices == inp_y)
        Variables.logger.info("Absolute validation accuracy is "+
                str(accurate.item())+"/"+str(datasetLengthValid)+"="+str(accurate.item()/datasetLengthValid * 100)+"%")
