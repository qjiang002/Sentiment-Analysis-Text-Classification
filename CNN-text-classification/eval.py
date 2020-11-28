#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from jieba import cut
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pickle
from gensim.models.keyedvectors import KeyedVectors
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/MR_polarity_5k/rt-polarity-test.pos", "Data source for the positive test data.")
tf.flags.DEFINE_string("negative_data_file", "./data/MR_polarity_5k/rt-polarity-test.neg", "Data source for the negative test data.")
tf.flags.DEFINE_string("model_type","rand","'rand' for CNN-rand; 'static' for CNN-static (default: rand)")
tf.flags.DEFINE_float("initialize_range", 0.2, "initialize range of word embedding")
tf.flags.DEFINE_string("dataset","COVID_Chinese","directory of dataset")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
def chinese_tokenizer(docs):
    for doc in docs:
        yield list(cut(doc))

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    if FLAGS.dataset == 'COVID_Chinese':
        x_raw, y_test = data_helpers.load_COVID_Chinese('../data/COVID_Chinese', train=False)
    elif FLAGS.dataset == 'SST-5':
        x_raw, y_test = data_helpers.load_SST5('../data/SST-5', train=False)
    else:
        x_raw, y_test = data_helpers.load_MR(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
if FLAGS.model_type=='rand':
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
else:
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab.pickle")
    with open(vocab_path, 'rb') as file:
        vocab_processor = pickle.load(file)
    print("type of vocab_processor: ", type(vocab_processor), len(vocab_processor))
    sentence_embedding = []
    vocab_processor['max_document_length'] = 54
    vocab_processor['embedding_dim'] = 300
    for x in x_raw:
        x = x.split(" ")
        single_sentence_embedding = []
        for word in x:
            if word in word2vec_model:
                single_sentence_embedding.append(word2vec_model[word])
            elif word in vocab_processor:
                single_sentence_embedding.append(vocab_processor[word])
            else:
                vocab_processor[word] = np.random.rand(vocab_processor['embedding_dim'])*FLAGS.initialize_range
                single_sentence_embedding.append(vocab_processor[word])
        if len(single_sentence_embedding) > vocab_processor['max_document_length']:
            single_sentence_embedding = single_sentence_embedding[:vocab_processor['max_document_length']]
        if len(single_sentence_embedding) < vocab_processor['max_document_length']:
            single_sentence_embedding.extend([np.zeros(vocab_processor['embedding_dim'])] * (vocab_processor['max_document_length'] - len(single_sentence_embedding)))
        single_sentence_embedding = np.array(single_sentence_embedding)
        sentence_embedding.append(single_sentence_embedding)
    x_test = np.array(sentence_embedding)
    print("x_test: ", x_test.shape, type(x_test))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
