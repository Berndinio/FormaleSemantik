from torch.utils.data import Dataset
import torch
from Variables import Variables

class RNNDataset(Dataset):
    def __init__(self, dSet, transform=None):
        self.dataSet = dSet

    def __len__(self):
        return self.dataSet.shape[0]

    def __getitem__(self, idx):
        # (allLength, self.word2VecDimensions, 1*2 + 2*maxLengthSentence + 1)
        sliced = self.dataSet[idx, :, -1][:81].type(torch.LongTensor)
        indice = sliced.argmax()
        return self.dataSet[idx, :, 2:2+43], indice

if __name__ == '__main__':
    face_dataset = RNNDataset()
