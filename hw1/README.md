# Intent Classification and Slot Tagging

## Description
- [link](https://docs.google.com/presentation/d/19FDqunmvGZMNVgQQ3Zb5iMOKp9TzPfi5C41Q858iGlw/edit#slide=id.gc74c89b633_0_13)

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent Classification
### Training
```shell
python train_intent.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device> --num_epoch <num_epoch> --seed <seed> --early_stop_step <early_stop_step> --model <model> --attention <attention> --head <head>
```
- **data_dir**: Directory to the dataset.
- **cache_dir**: Directory to the preprocessed caches.
- **ckpt_dir**: Directory to save the model file.
- **max_len**: vocabulary encode size.
- **hidden_size**: RNN hidden state dim.
- **num_layers**: Number of RNN layers.
- **dropout**: Model dropout rate.
- **bidirectional**: Do bidirectional or not.
- **lr**: Learning rate.
- **early_stop_step**: Early stop if model is not improving.
- **model**: Choose RNN model (rnn, lstm or gru). 
- **attention**: Do self-attention or not.
- **head**: Number of attention head.

### Inference
```shell
python test_intent.py --test_file <test_file> --cache_dir <cache_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --model <model> --attention <attention> --head <head>
```

- **test_file**: Path to the test file.
- **cache_dir**: Directory to the preprocessed caches.
- **ckpt_path**: Path to model checkpoint.
- **max_len**: vocabulary encode size.
- **hidden_size**: RNN hidden state dim.
- **num_layers**: Number of RNN layers.
- **dropout**: Model dropout rate.
- **bidirectional**: Do bidirectional or not.
- **model**: Choose RNN model (should be the same as training model). 
- **attention**: Do self-attention or not.
- **head**: Number of attention head.

### Reproduce
```shell
bash download.sh
bash intent_cls.sh {path_to_test_file} {path_to_predict_file.csv}
```

---

## Slot Tagging
### Training
```shell
python train_slot.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device> --num_epoch <num_epoch> --seed <seed> --early_stop_step <early_stop_step> --model <model> --seqeval <seqeval> --l1 <l1> --cnn <cnn>
```
- **data_dir**: Directory to the dataset.
- **cache_dir**: Directory to the preprocessed caches.
- **ckpt_dir**: Directory to save the model file.
- **max_len**: vocabulary encode size.
- **hidden_size**: RNN hidden state dim.
- **num_layers**: Number of RNN layers.
- **dropout**: Model dropout rate.
- **bidirectional**: Do bidirectional or not.
- **lr**: Learning rate.
- **early_stop_step**: Early stop if model is not improving.
- **model**: Choose RNN model (rnn, lstm or gru). 
- **seqeval**: Use seqeval framework to evaluate model.
- **l1**: Linear layer dimension.
- **cnn**: Do CNN-BiLSTM or not.

### Inference
```shell
python test_intent.py --test_file <test_file> --cache_dir <cache_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --model <model> --l1 <l1> --cnn <cnn> 
```

- **test_file**: Path to the test file.
- **cache_dir**: Directory to the preprocessed caches.
- **ckpt_path**: Path to model checkpoint.
- **max_len**: vocabulary encode size.
- **hidden_size**: RNN hidden state dim.
- **num_layers**: Number of RNN layers.
- **dropout**: Model dropout rate.
- **bidirectional**: Do bidirectional or not.
- **model**: Choose RNN model (should be the same as training model). 
- **l1**: Linear layer dimension.
- **cnn**: Do CNN-BiLSTM or not.

### Reproduce
```shell
bash download.sh
bash slot_tag.sh {path_to_test_file} {path_to_predict_file.csv}
```
