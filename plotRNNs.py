from os import walk
from Variables import Variables
import re
import matplotlib.pyplot as plt

def keySort(elem):
    s = elem
    filter = re.findall(".pt", s)
    if(len(filter)==0):
        return -1
    epoch = re.findall("\d+", s)
    epoch = epoch[1]
    dataset = re.findall("min\d+[-small]*", s)
    dataset = dataset[0]
    netType = re.findall("AttentionLSTM_[A-Z]*", s)
    netType = netType[0]
    listNettype = ["AttentionLSTM_NDNN", "AttentionLSTM_NN", "AttentionLSTM_ND"]
    listDataset = ["min15", "min5", "min2", "min5-small", "min2-small"]
    key = int(epoch) + (listNettype.index(netType)+1) * 100 + (listDataset.index(dataset)+1) * 1000
    return key

def preparePlot(s):
    x, y = [],[]
    dataset = re.findall("min\d+[-small]*", s)
    dataset = dataset[0]
    netType = re.findall("AttentionLSTM_[A-Z]*", s)
    netType = netType[0]
    return x,y,dataset,netType

if __name__ == '__main__':
    #sort the files
    f = []
    for (dirpath, dirnames, filenames) in walk("models"):
        f.extend(filenames)
        break
    f.sort(key=keySort)
    f = f[5:]
    for file in f:
        Variables.logger.debug(file)

    #begin plotting
    x,y,dataset,netType = preparePlot(f[0])
    for file in f:
        numbers = re.findall("\d+", file)
        epoch = int(numbers[1])
        if(epoch==0):
            #save old plot
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.plot(x,y)
            plt.axis([0, 100, 0, 100])
            plt.title(dataset+" "+netType)
            plt.savefig("plots/"dataset+"-"+netType+".png")
            #prepareNewPlot
            x,y,dataset,netType = preparePlot(file)
        x.append(epoch)
        y.append(numbers[-1])
    #save a last time
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(x,y)
    plt.axis([0, 100, 0, 100])
    plt.title(dataset+" "+netType)
    plt.savefig("plots/"dataset+"-"+netType+".png")
