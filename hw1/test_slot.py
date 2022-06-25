import json
import csv
import numpy as np
from tqdm import tqdm
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SlotTagDataset
from model import SlotTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SlotTagDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SlotTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(tag2idx),
        args.model,
        args.l1,
        args.cnn
    ).to(args.device)

    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    # TODO: predict dataset
    id_arr = []
    tag_arr = []
    len_arr = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            batch['tokens'] = batch['tokens'].to(args.device)
            pred = model(batch)
            _, pred_id = torch.max(pred, 1)
            pred_id = pred_id.cpu().tolist()
            id_arr += batch['id']
            tag_arr += pred_id
            len_arr += batch['len']

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tags'])
        for i, pred, tokens_len in zip(id_arr, tag_arr, len_arr):
            pred_tag_list = [dataset.idx2label(4 if tid == 9 else tid) for tid in pred[:tokens_len]]
            pred_tag_str = ' '.join(pred_tag_list)
            writer.writerow([i, pred_tag_str])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.tag.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--model", type=str, default='lstm')
    parser.add_argument("--l1", type=int, default=256)
    parser.add_argument("--cnn", type=bool, default=False)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
