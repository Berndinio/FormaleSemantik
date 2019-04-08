import json
import gensim
import pickle
import torch
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import re


from Variables import Variables

class DataProcessing:
    def __init__(self):
        Variables.logger.info("Loading Word2Vec")
        #throws KeyError if undefined
        #self.word2Vec = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        self.word2VecDimensions = 300
        Variables.logger.info("Finished Word2Vec loading")

        self.objectSubjects = []
        self.relations = []

    def loadJson(self, fName):
        f = open(fName,"r")
        contents = f.read()
        data = json.loads(contents)
        return data

    def generateSamples(self, rawData, stopIteration=100):
        processed = 0
        allLength = 0
        maxLength = 0
        unknownWords = set()
        unknownWordsCount = 0
        for key in rawData.keys():
            allLength += len(rawData[key])
            for sample in rawData[key]:
                if len(sample["tokens"])>maxLength:
                    maxLength = len(sample["tokens"])
        if stopIteration<allLength:
            allLength = stopIteration
        Variables.logger.info("Dataset size:"+str(allLength)+", Maximum sentence length:"+str(maxLength))
        finalSamples = torch.zeros(stopIteration, self.word2VecDimensions, maxLength+3)

        #process samples
        for key in rawData.keys():
            relation = key
            if relation not in self.relations:
                self.relations.append(relation)
            for sample in rawData[key]:
                processed +=1
                if(processed%1000 == 0):
                    Variables.logger.info("Processing sample "+str(processed)+"/"+str(allLength)+", #unknownWords:"+str(len(unknownWords))+", total unknownWords:"+str(len(unknownWordsCount)))
                tokens = " ".join(sample["tokens"])
                regex = re.compile("[0-9]", re.IGNORECASE)
                tokens = regex.sub("0", tokens)
                tokens = strip_punctuation(tokens)
                tokens = strip_multiple_whitespaces(tokens)
                tokens = tokens.lower()
                tokens = tokens.strip()

                subject = sample["h"]
                object = sample["t"]
                if subject not in self.objectSubjects:
                    self.objectSubjects.append(subject)
                if object not in self.objectSubjects:
                    self.objectSubjects.append(object)
                #add to the samples vector
                Variables.logger.debug(processed)
                finalSamples[processed-1,0,0] = self.objectSubjects.index(subject)
                finalSamples[processed-1,0,1] = self.objectSubjects.index(object)
                finalSamples[processed-1,0,maxLength+2] = self.relations.index(relation)
                for x, word in enumerate(tokens.split(" "), 0):
                    try:
                        vector = self.word2Vec[word]
                        finalSamples[processed-1,:,2+x] = torch.from_numpy(vector)
                    except KeyError:
                        unknownWords.add(word)
                        unknownWordsCount += 1
                    except:
                        Variables.logger.warning("Something bad happened, we dont know what!")
                if processed==stopIteration:
                    return finalSamples
        return finalSamples


    def saveSamples(self, samples, fName="dummy.pt"):
        torch.save(samples, fName)

    def loadSamples(self, fName="dummy.pt"):
        return torch.load(fName)


if __name__ == '__main__':


    proc = DataProcessing()
    rawData = proc.loadJson("data/fewrel_train.json")
    rawDataVal = proc.loadJson("data/fewrel_val.json")
    rawData.update(rawDataVal)
    samples = proc.generateSamples(rawData)
    proc.saveSamples(samples)
    data = proc.loadSamples()
    Variables.logger.debug(samples.shape)
    Variables.logger.debug(data.shape)
