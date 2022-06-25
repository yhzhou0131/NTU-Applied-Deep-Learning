from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len
import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batch = {}

        samples.sort(key = lambda x : len(x['text'].split()), reverse=True)
        batch['len'] = [len(s['text'].split()) for s in samples]

        batch['text'] = [sample['text'].split() for sample in samples]
        batch['text'] = self.vocab.encode_batch(batch['text'], self.max_len)
        batch['text'] = torch.tensor(batch['text'])
        
        if 'intent' in samples[0]:
            batch['intent'] = [self.label2idx(sample['intent']) for sample in samples]
            batch['intent'] = torch.tensor(batch['intent'])

        batch['id'] = [sample['id'] for sample in samples]

        return batch
        raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

    class SlotTagDataset(Dataset):
        def __init__(
            self,
            data: List[Dict],
            vocab: Vocab,
            label_mapping: Dict[str, int],
            max_len: int,
        ):
            self.data = data
            self.vocab = vocab
            self.label_mapping = label_mapping
            self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, index) -> Dict:
            instance = self.data[index]
            return instance

        @property
        def num_classes(self) -> int:
            return len(self.label_mapping)

        def collate_fn(self, samples: List[Dict]) -> Dict:
            # TODO: implement collate_fn
            batch = {}

            batch['len'] = [len(s['tokens']) for s in samples]

            batch['tokens'] = [s['tokens'] for s in samples]
            batch['tokens'] = self.vocab.encode_batch(batch['tokens'], self.max_len)
            batch['tokens'] = torch.tensor(batch['tokens'])
            
            if 'tags' in samples[0]:
                batch['tags'] = [[self.label2idx(tag) for tag in sample['tags']] for sample in samples]
                batch['tags'] = pad_to_len(batch['tags'], self.max_len, 4)
                batch['tags'] = torch.tensor(batch['tags'])

            batch['id'] = [sample['id'] for sample in samples]

            return batch
            raise NotImplementedError

        def label2idx(self, label: str):
            return self.label_mapping[label]

        def idx2label(self, idx: int):
            return self._idx2label[idx]