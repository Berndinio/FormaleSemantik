#https://www.researchgate.net/publication/328494683_FewRel_A_Large-Scale_Supervised_Few-Shot_Relation_Classification_Dataset_with_State-of-the-Art_Evaluation

import json


class Preprocessing:
    def __init__(self):
        pass
    def loadJson(self, fName):
        f = open(fName,"r")
        contents = f.read()
        data = json.loads(contents)
        return data

    def generateSamples(data, mode=0):
        pass

    def saveSamples(data):
        pass

    def loadSamples(data):
        pass


if __name__ == '__main__':
    prep = Preprocessing()
    data = prep.loadJson("data/fewrel_train.json")
    for key in data.keys():
        print(data[key][0])
        break;
