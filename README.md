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

To train the recurrent models tested, follow these steps:

* Download the dataset (train and validation) and save it to
"data/fewrel_train.json", "data/fewrel_val.json"

* Execute (**WARNING**: This needs some time and ~30GB of disk memory. 16GB of RAM are an advantage.):
```
python -m preprocessing -generate 1 -amount 999999 -prefix min15 -w2v min15
python -m preprocessing -generate 1 -amount 999999 -prefix min5 -w2v min5
python -m preprocessing -generate 1 -amount 999999 -prefix min2 -w2v min2
python -m preprocessing -generate 1 -amount 7000 -prefix min5-small -w2v mi5
python -m preprocessing -generate 1 -amount 7000 -prefix min2-small -w2v mi2
```
to create the datasets. The "-small" datasets have only 10 relations.

* Execute (**WARNING** This needs ~5GB of disk memory)(If there is no gpu argument given, the cpu is used):
```
python -m Trainers.RNNTrain -prefix min15 -network NDNN -gpu <GPU Index>
python -m Trainers.RNNTrain -prefix min15 -network ND -gpu <GPU Index>
python -m Trainers.RNNTrain -prefix min15 -network NN -gpu <GPU Index>
python -m Trainers.RNNTrain -prefix min5-small -network NDNN -gpu <GPU Index>
python -m Trainers.RNNTrain -prefix min5 -network NDNN -gpu <GPU Index>
python -m Trainers.RNNTrain -prefix min2-small -network NDNN -gpu <GPU Index>
python -m Trainers.RNNTrain -prefix min2 -network NDNN -gpu <GPU Index>
```
to train the models with different architectures on different datasets.

* Execute
```
python -m plotRNNs
```
to generate the classification accuracy plots.




## Convolutional models


## FC models
