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

from Variables import Variables

class DataProcessing:
    def __init__(self, fPrefix=None):
        Variables.logger.info("Loading Word2Vec")
        #throws KeyError if undefined
        self.word2Vec = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        self.word2VecDimensions = 300
        Variables.logger.info("Finished Word2Vec loading")

        if fPrefix is None:
            self.objectSubjects = ["None"]
            self.relations = ["None"]
            self.samples = None
        else:
            self.samples = torch.load("producedData/"+fPrefix+"-samples.pt")
            self.objectSubjects = torch.load("producedData/"+fPrefix+"-objectSubjects.pt")
            self.relations = torch.load("producedData/"+fPrefix+"-relations.pt")

        self.nlp = spacy.load("en_core_web_sm")

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
                path = []
                try:
                    path = nx.shortest_path(graph, source=subject.split(" ")[0], target=object.split(" ")[0])
                except:
                    Variables.logger.warning("No path found!")

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
                        #Variables.logger.warning("Something bad happened, we dont know what!")
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
                        #Variables.logger.warning("Something bad happened, we dont know what!")
                        pass
                #stop if already all processed
                if processed==stopIteration:
                    return self.samples
        return self.samples

    def fitTFIDF(self, rawData, stopIteration=100):
        from gensim.models import TfidfModel
        from gensim.corpora import Dictionary
        dataset = []
        processed = 0
        for key in rawData.keys():
            for sample in rawData[key]:
                processed += 1
                tokens = " ".join(sample["tokens"])
                tokens = self.polishSentence(tokens)
                dataset.append([tokens])
                if processed==stopIteration:
                    break
            if processed==stopIteration:
                break
        dct = Dictionary(dataset)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in dataset]
        model = TfidfModel(corpus)  # fit model
        vector = model[corpus[0]]  # apply model to the first corpus document
        vector2 = model[corpus[0]]  # apply model to the first corpus document
        Variables.logger.debug(dataset[0])
        Variables.logger.debug(vector2)
        Variables.logger.debug(vector)

    def saveSamples(self, samples, fPrefix="dummy"):
        torch.save(self.samples, "producedData/"+fPrefix+"-samples.pt")
        torch.save(self.objectSubjects, "producedData/"+fPrefix+"-objectSubjects.pt")
        torch.save(self.relations, "producedData/"+fPrefix+"-relations.pt")

    def loadSamples(self, fPrefix="dummy"):
        self.samples = torch.load("producedData/"+fPrefix+"-samples.pt")
        self.objectSubjects = torch.load("producedData/"+fPrefix+"-objectSubjects.pt")
        self.relations = torch.load("producedData/"+fPrefix+"-relations.pt")
        return samples


if __name__ == '__main__':
    proc = DataProcessing()
    rawData = proc.loadJson("data/fewrel_train.json")
    rawDataVal = proc.loadJson("data/fewrel_val.json")
    rawData.update(rawDataVal)

    samples = proc.generateSamples(rawData)
    proc.saveSamples(samples)
    loadedData = proc.loadSamples()

    #proc.fitTFIDF(rawData, 1000)
