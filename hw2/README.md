# Chinese Question Answering

## Description
- [link](https://docs.google.com/presentation/d/16QCzxSJoCRgx5ONlU8UU9H2qbKyRyGkSKzT9msEtuAc/edit#slide=id.gcf6a22f398_0_414)

## Environment
```shell
pip install -r requirement.txt
```

## Context Selection
### Training
```shell=
python multiple_choice.py \
  --do_train \
  --do_eval \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --overwrite_output \
  --context_file <path/to/context_file> \
  --train_file <path/to/train_file> \
  --validation_file  <path/to/validation_file> \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --pad_to_max_length True \
  --gradient_accumulation_steps 2 \
  --cache_dir ./cache \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2  \
  --warmup_ratio 0.1
```
- model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models. e.g. bert-base-chinese or hfl/chinese-macbert-larg
- output_dir: The output directory where the model predictions and checkpoints will be written. e.g. ./multiChoiceModel/bert_base or ./multiChoiceModel/macbert_large
- context_file: Path to the context file. EX: ./data/context.json
- train_file: path to training data file. EX: ./data/train.json
- validation_file: path to validation data file. EX: ./data/valid.json

### Testing
```shell=
python multiple_choice.py \
  --do_predict \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --context_file <path/to/context_file> \
  --test_file <path/to/test_file> \
  --max_seq_length 512 \
  --pad_to_max_length True \
  --cache_dir ./cache \
  --output_file  <path/to/output_file>
```
- model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models. e.g. ./multiChoiceModel/bert_base or ./multiChoiceModel/macbert_large
- output_dir: The output directory where the model predictions and checkpoints will be written. e.g. ./multiChoiceModel/bert_base or ./multiChoiceModel/macbert_large
- context_file: Path to the context file. EX: ./data/context.json
- test_file: Path to the testing file. EX: ./data/test.json
- output_file: Path to the output file. EX: ./multiChoicePredict.json

---
## Question Answering
### Training
```shell=
python run_qa.py \
  --do_train \
  --do_eval \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --context_file <path/to/context_file> \
  --train_file <path/to/train_file> \
  --validation_file  <path/to/validation_file> \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.1
```
- model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models. e.g. bert-base-chinese or hfl/chinese-macbert-larg
- output_dir: The output directory where the model predictions and checkpoints will be written. e.g. ./qaModel/bert_base or ./qaModel/macbert_large
- context_file: Path to the context file. EX: ./data/context.json
- train_file: path to training data file. EX: ./data/train.json
- validation_file: path to validation data file. EX: ./data/valid.json

### Testing
```shell=
python run_qa.py \
  --do_predict \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --test_file <path/to/test_file> \
  --context_file <path/to/context_file> \
  --output_file <path/to/output_file> \
  --max_seq_length 512 \
  --overwrite_output_dir \
  --overwrite_output  
```
- model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models. e.g. ./qaModel/bert_base or ./qaModel/macbert_large
- output_dir: The output directory where the model predictions and checkpoints will be written. e.g. ./qaModel/bert_base or ./qaModel/macbert_large
- context_file: Path to the context file. EX: ./data/context.json
- test_file: Path to the testing file. EX: ./data/test.json
- output_file: Path to the output file. EX: ./qaPrediction.json

## Reproduce my result
```shell=
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```
