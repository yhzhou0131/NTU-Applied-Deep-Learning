import json
import numpy as np
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 6, 7"
# torch.set_num_threads(2)

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloaders: Dict[str, SeqClsDataset] = {
        split: DataLoader(split_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn, pin_memory=True)
        for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(intent2idx),
        args.model,
        args.attention,
        args.head
    ).to(args.device)

    # TODO: init optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    step = 0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        train_acc = 0.0
        train_loss = 0.0
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for i, batch in enumerate(dataloaders[TRAIN]):
            batch['text'] = batch['text'].to(args.device)
            batch['intent'] = batch['intent'].to(args.device)

            # Forward pass
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch['intent'])

            # Backward and optimization
            loss.backward()
            optimizer.step()
            
            _, train_pred = torch.max(pred, 1)
            train_acc += (train_pred.detach() == batch['intent'].detach()).sum().item()
            train_loss += loss.item()        
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        val_acc = 0.0
        val_loss = 0.0

        if len(datasets[DEV]) > 0:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(dataloaders[DEV]):
                    batch['text'] = batch['text'].to(args.device)
                    batch['intent'] = batch['intent'].to(args.device)

                    pred = model(batch)
                    loss = criterion(pred, batch['intent'])

                    _, val_pred = torch.max(pred, 1)
                    val_acc += (val_pred.cpu() == batch['intent'].cpu()).sum().item()
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(epoch + 1, args.num_epoch, train_acc/len(datasets[TRAIN]), train_loss/len(dataloaders[TRAIN]), val_acc/len(datasets[DEV]), val_loss/len(dataloaders[DEV])))   

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), f'{args.ckpt_dir}/model.ckpt')    
                    print('Saving model with acc {:.3f}'.format(best_acc/len(datasets[DEV])))
                    step = 0
                else:
                    step += 1
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(epoch + 1, args.num_epoch, train_acc/len(datasets[TRAIN]), train_loss/len(dataloaders[TRAIN]))) 
        if step > args.early_stop_step:
            print('Model is not improving, so we halt the training session.')
            break    
        pass

    # TODO: Inference on test set
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=400)
    parser.add_argument("--seed", type=int, default=880131)
    parser.add_argument("--early_stop_step", type=int, default=60)
    parser.add_argument("--model", type=str, default='lstm')
    parser.add_argument("--attention", type=bool, default=False)
    parser.add_argument("--head", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    same_seed(args.seed)
    main(args)
