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
python -m preprocessing -generate 1 -amount 7000 -prefix min5-small -w2v min5
python -m preprocessing -generate 1 -amount 7000 -prefix min2-small -w2v min2
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
Papers read for the CNN models:

1. [Semantic Relation Classification via Convolutional Neural Networks with
Simple Negative Sampling](https://arxiv.org/abs/1506.07650)

2. [Relation Classification via Convolutional Deep Neural Network](https://www.aclweb.org/anthology/C14-1220) (main paper)

3. [Combining Recurrent and Convolutional Neural Networks
for Relation Classification](https://arxiv.org/abs/1605.07333)


To train the convolutional models, follow these steps:

* Download the dataset:
"python -m preprocessing -generate 1 -amount 56000 -prefix dummy"
(or change the amount value to train on only a part of the dataset)


*train,valuate and test CNNs
"python CNN.py"
(parameters can by changed directly in the CNN.py file)
As default this will train the data and saves a graph for loss and acc for the training AND validation for every epoch.

Default:
- CNN with input: sub,obj emembeddings

in this project we also trained a CNN with input: sub,obj,shortest dependency path emembeddings


## FC models
