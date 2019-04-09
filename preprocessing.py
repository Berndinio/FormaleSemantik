import json
import gensim
import pickle
import torch
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import re
import spacy
import networkx as nx
import numpy as np
import argparse

from Variables import Variables

class DataProcessing:
    def __init__(self, fPrefix=None):
        Variables.logger.info("Loading Word2Vec")
        #throws KeyError if undefined
        self.word2Vec = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        self.word2VecDimensions = 300
        Variables.logger.info("Finished Word2Vec loading")

        self.samples = None
        self.samplesSplitted = []
        self.processedDataset = []
        self.processedDatasetSplitted = []

        self.objectSubjects = ["None"]
        self.relations = ["None"]
        self.tfidfModel = None
        if fPrefix is not None:
            self.loadAll(fPrefix)

        self.nlp = spacy.load("en_core_web_sm")

    def loadJsonData(self, fPaths=["data/fewrel_train.json", "data/fewrel_val.json"]):
        data = {}
        for address in fPaths:
            rawData = proc.loadJson(address)
            data.update(rawData)
        return data

    def loadJson(self, fName):
        f = open(fName,"r")
        contents = f.read()
        data = json.loads(contents)
        return data

    def polishSentence(self, tokens):
        regex = re.compile("[0-9]", re.IGNORECASE)
        tokens = regex.sub("0", tokens)
        tokens = strip_punctuation(tokens)
        tokens = strip_multiple_whitespaces(tokens)
        tokens = tokens.lower()
        tokens = tokens.strip()
        return tokens

    def generateSamples(self, rawData, stopIteration):
        processed = 0
        allLength = 0
        maxLength = 0
        unknownWords = set()
        unknownWordsCount = 0
        processedDataset = []

        for key in rawData.keys():
            allLength += len(rawData[key])
            for sample in rawData[key]:
                if len(sample["tokens"])>maxLength:
                    maxLength = len(sample["tokens"])
        if stopIteration<allLength:
            allLength = stopIteration
        Variables.logger.info("Dataset size:"+str(allLength)+", Maximum sentence length:"+str(maxLength))
        self.samples = torch.zeros(stopIteration, self.word2VecDimensions, 2*maxLength+3)

        #process samples
        for key in rawData.keys():
            relation = key
            if relation not in self.relations:
                self.relations.append(relation)
            for sample in rawData[key]:
                processed +=1
                if(processed%1000 == 0):
                    Variables.logger.info("Processing sample "+str(processed)+"/"+str(allLength)+", #unknownWords:"+str(len(unknownWords))+", total unknownWords:"+str(unknownWordsCount))
                tokens = " ".join(sample["tokens"])
                tokens = self.polishSentence(tokens)

                subject = sample["h"][0]
                object = sample["t"][0]
                if subject not in self.objectSubjects:
                    self.objectSubjects.append(subject)
                if object not in self.objectSubjects:
                    self.objectSubjects.append(object)

                #create the parse tree for SDP
                edges = []
                doc = self.nlp(tokens)
                for token in doc:
                    if token.text!=token.head.text:
                        edges.append((token.text, token.head.text))
                graph = nx.Graph(edges)
                #find SDP
                path = []
                for s in subject.split(" "):
                    for o in object.split(" "):
                        try:
                            path = nx.shortest_path(graph, source=s, target=o)
                        except:
                            pass
                        if path!=[]:
                            break
                    if path!=[]:
                        break
                if path==[]:
                    Variables.logger.warning("No SDP found")

                #add subj/obj/rel to the samples vector
                self.samples[processed-1,0,0] = self.objectSubjects.index(subject)
                self.samples[processed-1,0,1] = self.objectSubjects.index(object)
                self.samples[processed-1,0,-1] = self.relations.index(relation)
                #
                #add sentence to the samples vector
                for x, word in enumerate(tokens.split(" "), 0):
                    try:
                        vector = self.word2Vec[word]
                        self.samples[processed-1,:,2+x] = torch.from_numpy(vector)
                    except KeyError:
                        unknownWords.add(word)
                        unknownWordsCount += 1
                    except:
                        Variables.logger.warning("Something bad happened, we dont know what!")
                        pass
                #
                #add SDP to the samples vector
                for x, word in enumerate(path,0):
                    try:
                        vector = self.word2Vec[word]
                        self.samples[processed-1,:,maxLength+2] = torch.from_numpy(vector)
                    except KeyError:
                        unknownWords.add(word)
                        unknownWordsCount += 1
                    except:
                        Variables.logger.warning("Something bad happened, we dont know what!")
                        pass
                self.processedDataset.append(tokens.split(" "))
                #stop if already all processed
                if processed==stopIteration:
                    return self.samples
        return self.samples

    def fitTFIDF(self, rawData, stopIteration):
        from gensim.models import TfidfModel
        from gensim.corpora import Dictionary
        dataset = []
        processed = 0
        for key in rawData.keys():
            for sample in rawData[key]:
                processed += 1
                tokens = " ".join(sample["tokens"])
                tokens = self.polishSentence(tokens)
                dataset.append(tokens.split(" "))
                if processed==stopIteration:
                    break
            if processed==stopIteration:
                break
        dct = Dictionary(dataset)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in dataset]
        self.tfidfModel = TfidfModel(corpus)  # fit model
        return self.tfidfModel

    def generateEverything(self, fPrefix="dummy", stopIteration=10000):
        #generate samples
        rawData = self.loadJsonData()
        self.generateSamples(rawData, stopIteration)
        #split samples
        train_size = int(0.7 * self.samples.shape[0])
        valid_size = int(0.2 * self.samples.shape[0])
        test_size = int(0.1 * self.samples.shape[0])
        randomNums = np.random.choice(self.samples.shape[0], self.samples.shape[0])
        trainRand = randomNums[:train_size]
        validRand = randomNums[train_size:train_size+valid_size]
        testRand = randomNums[train_size+valid_size:train_size+valid_size+test_size]
        train_samples = self.samples[trainRand]
        train_dataset = [self.processedDataset[i] for i in trainRand]
        valid_samples = self.samples[validRand]
        valid_dataset = [self.processedDataset[i] for i in validRand]
        test_samples = self.samples[testRand]
        test_dataset = [self.processedDataset[i] for i in testRand]
        self.processedDatasetSplitted = [train_samples, valid_dataset, test_dataset]
        train_samples = train_samples[None]
        valid_samples = valid_samples[None]
        test_samples = test_samples[None]
        self.samplesSplitted = [train_samples, valid_samples, test_samples]

        #train tfidf
        rawData = self.loadJsonData()
        self.fitTFIDF(rawData, stopIteration)

        #save things
        self.saveAll(fPrefix)

    def saveAll(self, fPrefix):
        #list with pytorch tensors [train_samples, valid_samples, test_samples]
        torch.save(self.samplesSplitted, "producedData/"+fPrefix+"-samplesSplitted.pt")
        #list with samples [train_samples, valid_samples, test_samples]
        torch.save(self.processedDatasetSplitted, "producedData/"+fPrefix+"-processedDatasetSplitted.pt")

        #just lists
        torch.save(self.objectSubjects, "producedData/"+fPrefix+"-objectSubjects.pt")
        torch.save(self.relations, "producedData/"+fPrefix+"-relations.pt")
        #just a pickled object
        torch.save(self.tfidfModel, "producedData/"+fPrefix+"-tfidfModel.pt")

    def loadAll(self, fPrefix):
        self.samplesSplitted = torch.load("producedData/"+fPrefix+"-samplesSplitted.pt")
        self.processedDatasetSplitted = torch.load("producedData/"+fPrefix+"-processedDatasetSplitted.pt")

        self.objectSubjects = torch.load("producedData/"+fPrefix+"-objectSubjects.pt")
        self.relations = torch.load("producedData/"+fPrefix+"-relations.pt")
        self.tfidfModel = torch.load("producedData/"+fPrefix+"-tfidfModel.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-generate", type=int, default=0, help="Nothing to see here")
    args = parser.parse_args()
    if(args.generate==1):
        Variables.logger.info("Generating the data")
        proc = DataProcessing()
        proc.generateEverything("10000")
    else:
        Variables.logger.info("Testing the data")
        proc = DataProcessing("10000")
        for s in proc.samplesSplitted:
            Variables.logger.debug(s.shape)
        for s in proc.processedDatasetSplitted:
            Variables.logger.debug(len(s))
