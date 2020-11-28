# BERT text classifier

* Download bert_tf from https://github.com/google-research/bert
* Download pre-trained bert models into file `checkpoint`: BERT-Base Uncased, BERT-Base Chinese
* Run `bert_classifier.py`
* Run `eval_matrix.py` to calculate accuracy

Example
```
python bert_classifier.py --data_dir=../data/MR_polarity_5k --bert_config_file=checkpoint/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt --vocab_file=checkpoint/uncased_L-12_H-768_A-12/vocab.txt --output_dir=./output/MR --max_seq_length 128 --do_train --do_eval --do_predict
```
