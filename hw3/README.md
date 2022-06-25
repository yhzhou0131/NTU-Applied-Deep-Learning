Natural Language Generation
===

# Description
- [link](https://docs.google.com/presentation/d/11pV5rM4-pxy7Aam5wZwaXHDNIuFEthdhlEXNXLDuWxc/edit?usp=sharing)

# Reproduce my results
```shell=
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

# Training
```shell=
python ./run_summarization.py \
    --model_name_or_path google/mt5-small \
    --output_dir ./eval-summarization \
    --do_train \
    --do_eval \
    --train_file /path/to/train_file \
    --validation_file /path/to/validation_file \
    --text_column maintext \
    --summary_column title \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --learning_rate 5e-4 \
    --optim adafactor \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --evaluation_strategy epoch 
```

# Inference
```shell=
python ./run_summarization.py \
    --model_name_or_path ./eval-summarization \
    --output_dir ./eval-summarization \
    --do_predict \
    --test_file /path/to/test_file \
    --output_file /path/to/output_file \
    --text_column maintext \
    --summary_column title \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --num_beams 5
```

# Dataset & evaluation script for ADL 2022 homework 3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Installation
```
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
```


## Usage
### Use the Script
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```


### Use Python Library
```
>>> from tw_rouge import get_rouge
>>> get_rouge('我是人', '我是一個人')
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], [ 我是一個人'])
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], ['我是一個人'], avg=False)
[{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}]
```


## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)
