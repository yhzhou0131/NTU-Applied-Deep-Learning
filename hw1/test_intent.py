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

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
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

    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    # TODO: predict dataset
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            batch['text'] = batch['text'].to(args.device)
            pred = model(batch)

            _, pred_id = torch.max(pred, 1)
            pred_id = pred_id.cpu().numpy()
            test_id = [int(id[5:]) for id in batch['id']]
            preds += [(testID, dataset.idx2label(id)) for id, testID in zip(pred_id, test_id)]
    preds.sort(key=lambda x: x[0])

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'intent'])
        for i, p in preds:
            writer.writerow([f'test-{i}', p])


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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--model", type=str, default='lstm')
    parser.add_argument("--attention", type=bool, default=False)
    parser.add_argument("--head", type=int, default=1)

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
