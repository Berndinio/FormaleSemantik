#According to paper https://www.aclweb.org/anthology/P16-2034

import torch
import torch.nn as nn
from Variables import Variables


class AttentionLSTM_NoDropoutNoNorm(nn.Module):
    def __init__(self):
        super(AttentionLSTM_NoDropoutNoNorm, self).__init__()
        self.hiddenSizeLSTM = 300
        self.maxSentenceLength = 43

        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hiddenSizeLSTM,
                                num_layers=1, dropout=0.0, bidirectional=True,
                                batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(300, 80)
        self.conv1d = nn.Conv1d(self.hiddenSizeLSTM, 1, 1,bias=False)
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)



    def forward(self, x):
        #samples x w2vecDim x seqLength
        batchSize = x.shape[0]
        hStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        cStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        #feed LSTM
        for idx in range(self.maxSentenceLength):
            out, hidden = self.lstm(x[:, None, :, idx].to(Variables.device), (hStates[idx].clone(), cStates[idx].clone()))
            hStates[idx+1] = hidden[0]
            cStates[idx+1] = hidden[1]
        hStates = torch.sum(hStates, 1)
        #now has size (self.maxSentenceLength+1, batchSize, self.hiddenSizeLSTM)
        hStates = hStates[1:]
        hStates = hStates.permute(1, 2, 0)

        # feed attention layer
        # NOW (batchSize, self.hiddenSizeLSTM, self.maxSentenceLength+1, 1*2)
        M = self.tanh(hStates)
        alpha = self.conv1d(M)
        alpha = self.softmax2(alpha)
        alpha = alpha.permute(0, 2, 1)
        r = torch.bmm(hStates, alpha)
        hStar = self.tanh(r)

        #finally classify with FC
        hStar = hStar.squeeze()
        out = self.fc(hStar)
        out = self.softmax1(out)
        return out

    def flatten(self, x):
        sh = x.shape
        finalshape = 1
        for i in sh:
            finalshape *= i
        return x.view(1, finalshape)


class AttentionLSTM_NoNorm(nn.Module):
    def __init__(self):
        super(AttentionLSTM_NoNorm, self).__init__()
        self.hiddenSizeLSTM = 300
        self.maxSentenceLength = 43

        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hiddenSizeLSTM,
                                num_layers=1, dropout=0.0, bidirectional=True,
                                batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(300, 80)
        self.conv1d = nn.Conv1d(self.hiddenSizeLSTM, 1, 1,bias=False)
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.5)


    def forward(self, x):
        #dropout
        x = self.drop1(x)

        #samples x w2vecDim x seqLength
        batchSize = x.shape[0]
        hStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        cStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        #feed LSTM
        for idx in range(self.maxSentenceLength):
            out, hidden = self.lstm(x[:, None, :, idx].to(Variables.device), (hStates[idx].clone(), cStates[idx].clone()))
            hStates[idx+1] = hidden[0]
            cStates[idx+1] = hidden[1]
        hStates = torch.sum(hStates, 1)
        #now has size (self.maxSentenceLength+1, batchSize, self.hiddenSizeLSTM)
        hStates = hStates[1:]
        hStates = hStates.permute(1, 2, 0)

        #dropout
        hStates = self.drop1(hStates)

        # feed attention layer
        # NOW (batchSize, self.hiddenSizeLSTM, self.maxSentenceLength+1, 1*2)
        M = self.tanh(hStates)
        alpha = self.conv1d(M)
        alpha = self.softmax2(alpha)
        alpha = alpha.permute(0, 2, 1)
        r = torch.bmm(hStates, alpha)
        hStar = self.tanh(r)

        #dropout
        hStar = self.drop1(hStar)

        #finally classify with FC
        hStar = hStar.squeeze()
        out = self.fc(hStar)
        out = self.softmax1(out)
        return out

    def flatten(self, x):
        sh = x.shape
        finalshape = 1
        for i in sh:
            finalshape *= i
        return x.view(1, finalshape)


class AttentionLSTM_NoDropout(nn.Module):
    def __init__(self):
        super(AttentionLSTM_NoDropout, self).__init__()
        self.hiddenSizeLSTM = 300
        self.maxSentenceLength = 43

        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hiddenSizeLSTM,
                                num_layers=1, dropout=0.0, bidirectional=True,
                                batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(300, 80)
        self.conv1d = nn.Conv1d(self.hiddenSizeLSTM, 1, 1,bias=False)
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)

        #self.drop1 = nn.Dropout(0.0)
        #self.drop2 = nn.Dropout(0.0)
        #self.drop3 = nn.Dropout(0.0)

        self.norm1 = nn.BatchNorm1d(300)


    def forward(self, x):
        #dropout
        #x = self.drop1(x)

        #samples x w2vecDim x seqLength
        batchSize = x.shape[0]
        hStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        cStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        #feed LSTM
        for idx in range(self.maxSentenceLength):
            out, hidden = self.lstm(x[:, None, :, idx].to(Variables.device), (hStates[idx].clone(), cStates[idx].clone()))
            hStates[idx+1] = hidden[0]
            cStates[idx+1] = hidden[1]
        hStates = torch.sum(hStates, 1)
        #now has size (self.maxSentenceLength+1, batchSize, self.hiddenSizeLSTM)
        hStates = hStates[1:]
        hStates = hStates.permute(1, 2, 0)

        #norm
        hStates = self.norm1(hStates)

        #dropout
        #hStates = self.drop1(hStates)

        # feed attention layer
        # NOW (batchSize, self.hiddenSizeLSTM, self.maxSentenceLength+1, 1*2)
        M = self.tanh(hStates)
        alpha = self.conv1d(M)
        alpha = self.softmax2(alpha)
        alpha = alpha.permute(0, 2, 1)
        r = torch.bmm(hStates, alpha)
        hStar = self.tanh(r)

        #dropout
        #hStar = self.drop1(hStar)

        #finally classify with FC
        hStar = hStar.squeeze()

        #norm
        hStar = self.norm1(hStar)

        out = self.fc(hStar)
        out = self.softmax1(out)
        return out

    def flatten(self, x):
        sh = x.shape
        finalshape = 1
        for i in sh:
            finalshape *= i
        return x.view(1, finalshape)




''' Some implementation approach for packing the paddings
class AttentionLSTM(nn.Module):
    def __init__(self):
        super(AttentionLSTM, self).__init__()
        self.hiddenSizeLSTM = 300
        self.maxSentenceLength = 43

        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hiddenSizeLSTM,
                                num_layers=1, dropout=0.0, bidirectional=True,
                                batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(300, 80)
        self.conv1d = nn.Conv1d(self.hiddenSizeLSTM, 1, 1,bias=False)
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.5)


    def forward(self, x):
        batchSize = x.shape[0]

        #pack padding
        #get lengths
        tempx = torch.sum(x, dim=1)
        lengths = torch.ones((batchSize))
        for idx, ele in enumerate(x):
            length = tempx[idx, :].nonzero()[-1,0]+1
            lengths[idx] = length
        #sort by length
        sortedIndices = torch.sort(lengths, dim=0, descending=True)[1]
        Variables.logger.debug(x.shape)
        Variables.logger.debug(lengths.shape)
        x = x[sortedIndices]
        lengths = lengths[sortedIndices]
        lengths = lengths.type(torch.int32)

        #dropout
        x = self.drop1(x)

        #pack padded
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        #samples x w2vecDim x seqLength
        hStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        cStates = torch.zeros((self.maxSentenceLength+1, 1*2, batchSize, self.hiddenSizeLSTM)).to(Variables.device)
        #feed LSTM
        for idx in range(self.maxSentenceLength):
            out, hidden = self.lstm(x[:, None, :, idx].to(Variables.device), (hStates[idx].clone(), cStates[idx].clone()))
            hStates[idx+1] = hidden[0]
            cStates[idx+1] = hidden[1]
        hStates = torch.sum(hStates, 1)
        #now has size (self.maxSentenceLength+1, batchSize, self.hiddenSizeLSTM)
        hStates = hStates[1:]
        hStates = hStates.permute(1, 2, 0)

        #dropout
        hStates = self.drop1(hStates)

        # feed attention layer
        # NOW (batchSize, self.hiddenSizeLSTM, self.maxSentenceLength+1, 1*2)
        M = self.tanh(hStates)
        alpha = self.conv1d(M)
        alpha = self.softmax2(alpha)
        alpha = alpha.permute(0, 2, 1)
        r = torch.bmm(hStates, alpha)
        hStar = self.tanh(r)

        #dropout
        hStar = self.drop1(hStar)

        #finally classify with FC
        hStar = hStar.squeeze()
        out = self.fc(hStar)
        out = self.softmax1(out)
        return out

    def flatten(self, x):
        sh = x.shape
        finalshape = 1
        for i in sh:
            finalshape *= i
        return x.view(1, finalshape)'''



if __name__ == '__main__':
    input = torch.ones((10, 300, 43))
    model = AttentionLSTM()
    model(input)
