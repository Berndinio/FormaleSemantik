from os import walk
from Variables import Variables
import re
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import numpy as np

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
    x, y = [0],[0]
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
    x = []
    y = []
    for file in f:
        numbers = re.findall("\d+", file)
        epoch = int(numbers[1])
        if(epoch==0):
            #save old plot
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.ylim(0,100)
            plt.plot(x, y, label=dataset+" "+netType)
            plt.title(dataset+" "+netType)
            plt.savefig("plots/"+dataset+"_"+netType+".png")
            plt.clf()
            #prepareNewPlot
            x,y,dataset,netType = preparePlot(file)
            Variables.logger.debug("New plot"+dataset+" "+netType)
        x.append(epoch)
        y.append(int(numbers[-1]))
    #save a last time
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.ylim(0,100)
    plt.plot(x, y, label=dataset+" "+netType)
    plt.title(dataset+" "+netType)
    plt.savefig("plots/"+dataset+"_"+netType+".png")





    #begin plotting
    fig = plt.figure()
    ax = plt.subplot(111)
    x,y,dataset,netType = preparePlot(f[0])
    x = []
    y = []
    first = True
    for file in f:
        numbers = re.findall("\d+", file)
        epoch = int(numbers[1])
        if(epoch==0):
            if not first:
                #save old plot
                plt.ylabel("Accuracy")
                plt.xlabel("Epoch")
                plt.ylim(0,100)
                ax.plot(x, y, label=dataset+" "+netType)
            #prepareNewPlot
            x,y,dataset,netType = preparePlot(file)
            Variables.logger.debug("New plot"+dataset+" "+netType)
            first = False
        x.append(epoch)
        y.append(int(numbers[-1]))
    #save a last time
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.ylim(0,100)
    ax.plot(x, y, label=dataset+" "+netType)

    # Shrink current axis by 50%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height * 0.5])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title("All")
    plt.savefig("plots/All.png", dpi=600)
