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
        self.word2Vec = gensim.models.KeyedVectors.load_word2vec_format('models/enwiki.skip.size300.win10.neg15.sample1e-5.min15.bin', binary=True)
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
        maxLengthSentence = 0
        maxLengthSubject = 0
        maxLengthObject = 0
        unknownWords = set()
        unknownWordsCount = 0
        processedDataset = []

        for key in rawData.keys():
            for sample in rawData[key]:
                allLength += 1
                if len(self.polishSentence(" ".join(sample["tokens"])).split(" "))>maxLengthSentence:
                    maxLengthSentence = len(self.polishSentence(" ".join(sample["tokens"])).split(" "))
                if len(self.polishSentence(sample["h"][0]).split(" "))>maxLengthSubject:
                    maxLengthSubject = len(self.polishSentence(sample["h"][0]).split(" "))
                if len(self.polishSentence(sample["t"][0]).split(" "))>maxLengthObject:
                    maxLengthObject = len(self.polishSentence(sample["t"][0]).split(" "))
        Variables.logger.info("Dataset size:"+str(allLength)+", Maximum sentence length:"+str(maxLengthSentence)
                    +", Maximum subject length:"+str(maxLengthSubject)
                    +", Maximum object length:"+str(maxLengthObject))
        if stopIteration<allLength:
            allLength = stopIteration
        self.samples = torch.zeros(allLength, self.word2VecDimensions, 1*2 + 2*maxLengthSentence + 1)
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
                subject = self.polishSentence(subject)
                object = sample["t"][0]
                object = self.polishSentence(object)

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

                #add subject
                vector = torch.zeros((300,))
                vectorLength = 0
                for x, word in enumerate(subject.split(" ")):
                    try:
                        vector_tmp = self.word2Vec[word]
                        vector += torch.from_numpy(vector_tmp)
                        vectorLength += 1
                    except KeyError:
                        #Variables.logger.debug("Subject Word not present")
                        pass
                    except Exception as e:
                        Variables.logger.error(e)
                if vectorLength!=0:
                    self.samples[processed-1,:,0] = vector/vectorLength
                else:
                    Variables.logger.debug("Subject not present")

                #add object
                vector = torch.zeros((300,))
                vectorLength = 0
                for x, word in enumerate(object.split(" ")):
                    try:
                        vector_tmp = self.word2Vec[word]
                        vector += torch.from_numpy(vector_tmp)
                        vectorLength += 1
                    except KeyError:
                        #Variables.logger.debug("Object Word not present")
                        pass
                    except Exception as e:
                        Variables.logger.error(e)
                if vectorLength!=0:
                    self.samples[processed-1,:,1] = vector/vectorLength
                else:
                    Variables.logger.debug("Object not present")
                #add relation
                self.samples[processed-1, self.relations.index(relation), -1] = 1.0

                #
                #add sentence to the samples vector
                counter = 0
                for x, word in enumerate(tokens.split(" "), 0):
                    try:
                        vector = self.word2Vec[word]
                        self.samples[processed-1,:,2+counter] = torch.from_numpy(vector)
                        counter += 1
                    except KeyError:
                        unknownWords.add(word)
                        unknownWordsCount += 1
                    except Exception as e:
                        Variables.logger.error(e)
                #
                #add SDP to the samples vector
                counter = 0
                for x, word in enumerate(path,0):
                    try:
                        vector = self.word2Vec[word]
                        self.samples[processed-1,:,maxLengthSentence+2+counter] = torch.from_numpy(vector)
                        counter += 1
                    except KeyError:
                        unknownWords.add(word)
                        unknownWordsCount += 1
                    except Exception as e:
                        Variables.logger.error(e)
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

    def generateEverything(self, fPrefix="dummy", stopIteration=10):
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
        self.processedDatasetSplitted = [train_dataset, valid_dataset, test_dataset]
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
    parser.add_argument("-amount", type=int, default=2*700, help="Nothing to see here")
    parser.add_argument("-prefix", type=str, default="dummy", help="Nothing to see here")
    args = parser.parse_args()
    if(args.generate==1):
        Variables.logger.info("Generating the data")
        proc = DataProcessing()
        proc.generateEverything(args.prefix, args.amount)
    else:
        Variables.logger.info("Testing the data")
        proc = DataProcessing(args.prefix)
        for s in proc.samplesSplitted:
            Variables.logger.debug(s.shape)
        for s in proc.processedDatasetSplitted:
            Variables.logger.debug(len(s))
