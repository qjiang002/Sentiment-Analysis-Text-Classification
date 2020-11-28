# Text Classification (Sentiment Analysis)

## Datasets
Three datasets of different fine-grained levels in both English and Chinese.
* [MR](http://www.cs.cornell.edu/people/pabo/movie-review-data/): It is an English Movie Review dataset with one-sentence review per movie. The classification includes two classes: positive and negative.
* [COVID_Chinese](https://www.datafountain.cn/competitions/423/datasets): It is a Chinese dataset consisted of Weibo posts during COVID-19 from st th
January 1 to February 20 . It is a multimodal dataset with text, pictures and videos, but only text was used in this project. The classification includes three classes: positive, neutral and negative.
* [SST-5](https://nlp.stanford.edu/sentiment/): It is an English fine-grained sentiment classification dataset from Stanford Sentiment Treebank. Data is provided at phrase-level, so the sentences after data transformation were used for training and testing. The classification includes five classes: very negative, negative, neutral, positive and very positive.


## Classifiers
* TextCNN: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
* RNN
* CharCNN: [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)
* Very Deep CNN (VDCNN): [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)
* Bi-LSTM
* Attention-Based Bi-LSTM: [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)
* RCNN: [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
* BERT
