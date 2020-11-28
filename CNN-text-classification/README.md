# TextCNN classifier

## Training

```
python train.py
```
Parameters:
```
  --allow_soft_placement: Allow device soft device placement
  --batch_size: Batch Size (default: 50)
  --checkpoint_every: Save model after this many steps (default: 100)
  --dataset: directory of dataset: COVID_Chinese/SST-5/MR
  --dev_sample_percentage: Percentage of the training data to use for validation
  --dropout_keep_prob: Dropout keep probability (default: 0.5)
  --embedding_dim: Dimensionality of character embedding (default: 300)
  --evaluate_every: Evaluate model on dev set after this many steps (default: 100)
  --filter_sizes: Comma-separated filter sizes (default: '3,4,5')
  --initialize_range: initialize range of word embedding
  --l2_reg_lambda: L2 regularization lambda (default: 0.0)
  --learning_rate: Which learning rate to start with. (Default: 1e-3)
  --log_device_placement: Log placement of ops on devices
  --max_sentence_length: Max sentence length in train/test data (Default: 100)
  --model_type: 'rand' for CNN-rand; 'static' for CNN-static (default: rand)
  --negative_data_file: Data source for the MR negative data.
  --num_checkpoints: Number of checkpoints to store (default: 5)
  --num_epochs: Number of training epochs (default: 200)
  --num_filters: Number of filters per filter size (default: 100)
  --positive_data_file: Data source for the MR positive data.
  --word2vec: Word2vec file with pre-trained embeddings. (Default: None)

```


## Evaluating

```
python eval.py --eval_train --checkpoint_dir=./runs/1603201202/checkpoints --dataset=MR
```

Replace the checkpoint dir with the output from the training. Use `--eval_train` to evaluate the training datasets.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)

