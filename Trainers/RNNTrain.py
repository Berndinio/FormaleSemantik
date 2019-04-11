from preprocessing import DataProcessing
from DatasetClasses.RNNDataset import RNNDataset
from models.RNNModel import AttentionLSTM
from torch.utils.data import Dataset, DataLoader
from Variables import Variables

import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    proc = DataProcessing("dummy")
    train_dataset = RNNDataset(proc.samplesSplitted[0])

    net = AttentionLSTM()
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    for i_batch, (inp_x, inp_y) in enumerate(dataloader):
        Variables.logger.debug(i_batch)
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(inp_x)
        loss = criterion(output, inp_y)
        loss.backward()
        optimizer.step()    # Does the update
