import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Variables import Variables
from torch.utils.data import Dataset, DataLoader
import argparse
from preprocessing import DataProcessing
import sys


class FullyConnectedNet(nn.Module):

    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(12900, 5000)
        self.fc2 = nn.Linear(5000, 2500)
        self.fc3 = nn.Linear(2500, 1250)
        self.fc4 = nn.Linear(1250, 600)
        self.fc5 = nn.Linear(600, 100)
        self.fc6 = nn.Linear(100, 10)
        self.sm = nn.Softmax(dim=2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return self.sm(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FCDataset(Dataset):
    def __init__(self, dSet, transform=None):
        self.dataSet = dSet

    def __len__(self):
        return self.dataSet.shape[0]

    def __getitem__(self, idx):
        sliced = self.dataSet[idx, :, -1][:80].type(torch.LongTensor)
        indice = sliced.argmax()
        data = self.dataSet[idx, :, 2:2+43]
        return data.contiguous().view(1, data.size()[0]*data.size()[1]), indice - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", type=str, default="dummy", help="Nothing to see here")
    parser.add_argument("-gpu", type=int, default=0, help="Nothing to see here")
    args = parser.parse_args()
    if Variables.device is None:
        Variables.device = torch.device('cuda', args.gpu)

    batch_size = 10
    num_epochs = 10
    proc = DataProcessing(args.prefix, loadw2v=False)
    train_dataset = FCDataset(proc.samplesSplitted[0])
    valid_dataset = FCDataset(proc.samplesSplitted[1])
    test_dataset = FCDataset(proc.samplesSplitted[2])

    Variables.logger.info("Loading FC Network")
    net = FullyConnectedNet()
    Variables.logger.info("Network loaded")
    net = net.to(Variables.device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.5)
    criterion = nn.NLLLoss()
    datasetLength = len(train_dataloader)
    datasetLengthValid = len(valid_dataset)
    for epoch in range(num_epochs):
        net.train()
        for i_batch, (inp_x, inp_y) in enumerate(train_dataloader):
            inp_x = inp_x.to(Variables.device)
            inp_y = inp_y.to(Variables.device)
            # compute progress
            progress = int(i_batch/datasetLength * 100)
            sys.stdout.write("\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", Progress: "+'{:>3}'.format(str(progress))+"%")
            sys.stdout.flush()

            optimizer.zero_grad()   # zero the gradient buffers
            output = net(inp_x)
            output = output.squeeze()
            loss = criterion(output, inp_y)
            loss.backward()
            optimizer.step()    # Does the update
        sys.stdout.write("\rEpoch "+str(epoch+1)+"/"+str(num_epochs)+", Progress:  100%")
        print(" ")

        # compute valid accuracy
        net.eval()
        accurate = 0
        for i_batch, (inp_x, inp_y) in enumerate(valid_dataloader):
            inp_x = inp_x.to(Variables.device)
            inp_y = inp_y.to(Variables.device)
            output = net(inp_x)
            output = output.squeeze()
            indices = output.argmax(dim=1)

            accurate += torch.sum(indices == inp_y)
        Variables.logger.info("Absolute validation accuracy is "+
                str(accurate.item())+"/"+str(datasetLengthValid)+"="+str(accurate.item()/datasetLengthValid * 100)+"%")
        torch.save(net, "models/"+args.prefix+"_"
            + "FCN_"+"-epoch_"
            +str(epoch)+"-correct_"
            +str(int(accurate.item()/datasetLengthValid * 100))
            +".pt")