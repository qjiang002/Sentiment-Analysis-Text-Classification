import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np
from jieba import cut

TRAIN_PATH = "dbpedia_csv/train.csv"
TEST_PATH = "dbpedia_csv/test.csv"


def download_dbpedia():
    dbpedia_url = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'

    wget.download(dbpedia_url)
    with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
        tar.extractall()


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def build_word_dict(dataset, max_sentence_length):
    
    if not os.path.exists(dataset+"_word_dict.pickle"):
        print("building word_dict...")
        contents = []
        if dataset == 'MR_polarity_5k':
            with open('../data/MR_polarity_5k/train.txt', 'r') as f:
                lines = f.readlines()
                contents.extend([line.strip().split('\t')[0] for line in lines if len(line.strip().split('\t'))==2])
            with open('../data/MR_polarity_5k/dev.txt', 'r') as f:
                lines = f.readlines()
                contents.extend([line.strip().split('\t')[0] for line in lines if len(line.strip().split('\t'))==2])
        elif dataset == 'SST-5':
            with open('../data/SST-5/sst_train.txt', 'r') as f:
                lines = f.readlines()
                contents.extend([line.strip().split('\t')[1] for line in lines if len(line.strip().split('\t'))==2])
            with open('../data/SST-5/sst_dev.txt', 'r') as f:
                lines = f.readlines()
                contents.extend([line.strip().split('\t')[1] for line in lines if len(line.strip().split('\t'))==2])
        elif dataset == 'COVID_Chinese':
            with open('../data/COVID_Chinese/train.txt', 'r') as f:
                lines = f.readlines()
                contents.extend([line.strip().split('\t')[0] for line in lines if len(line.strip().split('\t'))==2])
            with open('../data/COVID_Chinese/dev.txt', 'r') as f:
                lines = f.readlines()
                contents.extend([line.strip().split('\t')[0] for line in lines if len(line.strip().split('\t'))==2])
        
        words = []
        if dataset == 'COVID_Chinese':
            for content in contents:
                words.extend(list(cut(content)))
        else:
            for content in contents:
                for word in word_tokenize(clean_str(content)):
                    words.append(word)
        
        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open(dataset+"_word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        print("load word_dict...")
        with open(dataset+"_word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)
    print("finish word_dict")
    return word_dict


def build_word_dataset(dataset, step, word_dict, document_max_len):
    text = []
    label = []
    if dataset == 'MR_polarity_5k':
        if step == "train":
            with open('../data/MR_polarity_5k/train.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[0])
                        if line[1]=='-1':
                            label.append(0)
                        else:
                            label.append(1)
            with open('../data/MR_polarity_5k/dev.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[0])
                        if line[1]=='-1':
                            label.append(0)
                        else:
                            label.append(1)
        else:
            with open('../data/MR_polarity_5k/test.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[0])
                        if line[1]=='-1':
                            label.append(0)
                        else:
                            label.append(1)
    elif dataset == 'SST-5':
        if step == "train":
            with open('../data/SST-5/sst_train.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[1])
                        label.append(int(line[0].replace('__label__',''))-1)
            with open('../data/SST-5/sst_dev.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[1])
                        label.append(int(line[0].replace('__label__',''))-1)
        else:
            with open('../data/SST-5/sst_test.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[1])
                        label.append(int(line[0].replace('__label__',''))-1)
    elif dataset == 'COVID_Chinese':
        if step == "train":
            with open('../data/COVID_Chinese/train.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2 and line[1] in ['-1', '0', '1']:
                        text.append(line[0])
                        label.append(int(line[1])+1)
            with open('../data/COVID_Chinese/dev.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2 and line[1] in ['-1', '0', '1']:
                        text.append(line[0])
                        label.append(int(line[1])+1)
        else:
            with open('../data/COVID_Chinese/test.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2 and line[1] in ['-1', '0', '1']:
                        text.append(line[0])
                        label.append(int(line[1])+1)
    
    if dataset == 'COVID_Chinese':
        x = list(map(lambda d: list(cut(d)), text))
    else:
        x = list(map(lambda d: word_tokenize(clean_str(d)), text))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    
    return x, label


def build_char_dataset(dataset, step, model, document_max_len):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    text = []
    label = []
    if dataset == 'MR_polarity_5k':
        if step == "train":
            with open('../data/'+dataset+'/train.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[0])
                        if line[1]=='-1':
                            label.append(0)
                        else:
                            label.append(1)
            with open('../data/'+dataset+'/dev.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[0])
                        if line[1]=='-1':
                            label.append(0)
                        else:
                            label.append(1)
        else:
            with open('../data/'+dataset+'/test.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[0])
                        if line[1]=='-1':
                            label.append(0)
                        else:
                            label.append(1)
    elif dataset == 'SST-5':
        if step == "train":
            with open('../data/'+dataset+'/sst_train.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[1])
                        label.append(int(line[0].replace('__label__',''))-1)
            with open('../data/'+dataset+'/sst_dev.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[1])
                        label.append(int(line[0].replace('__label__',''))-1)
        else:
            with open('../data/'+dataset+'/sst_test.txt') as f:
                for line in f:
                    line =  line.strip().split('\t')
                    if len(line)==2:
                        text.append(line[1])
                        label.append(int(line[0].replace('__label__',''))-1)
    # Shuffle dataframe
    #df = df.sample(frac=1)

    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)

    alphabet_size = len(alphabet) + 2

    x = list(map(lambda content: list(map(lambda d: char_dict.get(d, char_dict["<unk>"]), content.lower())), text))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]], x))

    #y = list(map(lambda d: d - 1, list(df["class"])))
    return x, label, alphabet_size


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    #print("inputs shape: ", inputs.shape)
    #print("outputs shape: ", outputs.shape)
    #print(inputs[0:5])
    #print(outputs[0:5])
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
