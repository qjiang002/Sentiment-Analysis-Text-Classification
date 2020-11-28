import argparse
import tensorflow as tf
from data_utils import *
from sklearn.model_selection import train_test_split
from cnn_models.word_cnn import WordCNN
from cnn_models.char_cnn import CharCNN
from cnn_models.vd_cnn import VDCNN
from rnn_models.word_rnn import WordRNN
from rnn_models.attention_rnn import AttentionRNN
from rnn_models.rcnn import RCNN


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="word_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
parser.add_argument("--dataset", type=str, default="MR_polarity_5k",
                    help="MR_polarity_5k | SST-5 | COVID_Chinese")
parser.add_argument("--num_class", type=int, default=2,
                    help="MR_polarity_5k: 2 | SST-5: 5 | COVID_Chinese: 3")
parser.add_argument("--num_epochs", type=int, default=100,
                    help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=2e-5,
                    help="learning rate")
parser.add_argument("--l2_reg_lambda", type=float, default=4.0,
                    help="l2_reg_lambda")
args = parser.parse_args()

#if not os.path.exists("dbpedia_csv"):
#    print("Downloading dbpedia dataset...")
#    download_dbpedia()

NUM_CLASS = args.num_class
BATCH_SIZE = 32
NUM_EPOCHS = args.num_epochs
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014
LEARNING_RATE = args.learning_rate
L2_REG_LAMBDA = args.l2_reg_lambda
DATASET = args.dataset

print("Building dataset...")
if args.model == "char_cnn":
    x, y, alphabet_size = build_char_dataset(DATASET, "train", "char_cnn", CHAR_MAX_LEN)
elif args.model == "vd_cnn":
    x, y, alphabet_size = build_char_dataset("train", "vdcnn", CHAR_MAX_LEN)
else:
    word_dict = build_word_dict(DATASET, WORD_MAX_LEN)
    vocabulary_size = len(word_dict)
    x, y = build_word_dataset(DATASET, "train", word_dict, WORD_MAX_LEN)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.1)


with tf.Session() as sess:
    if args.model == "word_cnn":
        model = WordCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    elif args.model == "char_cnn":
        model = CharCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS, LEARNING_RATE, L2_REG_LAMBDA)
    elif args.model == "vd_cnn":
        model = VDCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)
    elif args.model == "word_rnn":
        model = WordRNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS, LEARNING_RATE, L2_REG_LAMBDA)
    elif args.model == "att_rnn":
        model = AttentionRNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS, LEARNING_RATE, L2_REG_LAMBDA)
    elif args.model == "rcnn":
        model = RCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS, LEARNING_RATE, L2_REG_LAMBDA)
    else:
        raise NotImplementedError()

    checkpoint_dir = os.path.abspath(os.path.join(args.model, DATASET))
    checkpoint_prefix = os.path.join(checkpoint_dir, args.model)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }

        _, step, loss, acc = sess.run([model.optimizer, model.global_step, model.loss, model.accuracy], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}, acc = {2}".format(step, loss, acc))

        if step % 2000 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_loss, sum_accuracy, cnt = 0, 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }

                loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                sum_loss += loss
                cnt += 1
            valid_accuracy = sum_accuracy / cnt

            print("\nValidation Accuracy = {1}, Validation Loss = {2}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt, sum_loss / cnt))

            # Save model
            saver.save(sess, checkpoint_prefix, global_step=step)
            print("Model is saved.\n")
