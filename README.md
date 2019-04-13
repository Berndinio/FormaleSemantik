# FormaleSemantik
**Every folder must contain a __init__.py**.

Start programs in root folder with
```
python -m <filename without .py>
```
For example if you want to start the preprocessing.py ```if __name__ == '__main__':``` part, just type:
```
python -m preprocessing
```

For example if you want to start the model/preprocessing.py ```if __name__ == '__main__':``` part, just type:
```
python -m model.preprocessing
```


## Dataset
[FewRel](https://www.researchgate.net/publication/328494683_FewRel_A_Large-Scale_Supervised_Few-Shot_Relation_Classification_Dataset_with_State-of-the-Art_Evaluation) Dataset

## Word2Vec
Pretrained Word2Vec model can be downloaded [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).

**Additionally** we trained our own word2vec models on Wikidata with
[this](https://github.com/jind11/word2vec-on-wikipedia) repo
, because many words were unknown in the pretrained model above.
The [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html#download) server will be needed to use this trainer. Start it with
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## Recurrent models
Papers read for the RNN models:

1. [Classifying Relations via Long Short Term Memory Networksalong Shortest Dependency Paths](https://www.aclweb.org/anthology/D15-1206)

2. [Attention-Based Bidirectional Long Short-Term Memory Networks forRelation Classification](https://www.aclweb.org/anthology/P16-2034) (main paper)

3. [Relation Classification via Recurrent Neural Network](https://arxiv.org/pdf/1508.01006.pdf)

4. [Improved Relation Classification by Deep Recurrent Neural Networks with Data Augmentation](https://arxiv.org/pdf/1601.03651.pdf)

## Convolutional models


## FC models
