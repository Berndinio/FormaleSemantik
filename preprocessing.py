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
    def __init__(self, fPrefix=None):
        Variables.logger.info("Loading Word2Vec")
        #throws KeyError if undefined
        #self.word2Vec = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        self.word2VecDimensions = 300
        Variables.logger.info("Finished Word2Vec loading")

        if fPrefix is None:
            self.objectSubjects = []
            self.relations = []
            self.samples = None
        else:
            self.objectSubjects = torch.save(samples, fPrefix+"-samples.pt")
            self.relations = torch.save(samples, fPrefix+"-objectSubjects.pt")
            self.samples = torch.save(samples, fPrefix+"-relations.pt")

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
        self.samples = finalSamples
        return finalSamples


    def saveSamples(self, samples, fPrefix="dummy"):
        torch.save(samples, "producedData/"+fPrefix+"-samples.pt")
        torch.save(samples, "producedData/"+fPrefix+"-objectSubjects.pt")
        torch.save(samples, "producedData/"+fPrefix+"-relations.pt")

    def loadSamples(self, fPrefix="dummy"):
        self.objectSubjects = torch.save(samples, "producedData/"+fPrefix+"-samples.pt")
        self.relations = torch.save(samples, "producedData/"+fPrefix+"-objectSubjects.pt")
        self.samples = torch.save(samples, "producedData/"+fPrefix+"-relations.pt")
        return samples


if __name__ == '__main__':
    proc = DataProcessing()
    rawData = proc.loadJson("data/fewrel_train.json")
    rawDataVal = proc.loadJson("data/fewrel_val.json")
    rawData.update(rawDataVal)
    samples = proc.generateSamples(rawData)
    proc.saveSamples(samples)
    loadedData = proc.loadSamples()
