from ctypes import ArgumentError
from typing import Dict

import torch
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        model: str,
        attention: bool,
        head: int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embedding_dim = embeddings.size(dim=1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.model = model
        self.attention = attention
        self.head = head
        
        self.rnn = torch.nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout, 
            bidirectional=self.bidirectional,
            batch_first = True
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout, 
            bidirectional=self.bidirectional,
            batch_first = True
        )
        self.gru = torch.nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout, 
            bidirectional=self.bidirectional,
            batch_first = True
        )
        self.hidden2out = torch.nn.Sequential(
            torch.nn.Dropout(0.7),
            torch.nn.Linear(self.encoder_output_size, self.num_class),
            # torch.nn.Linear(self.hidden_size, self.num_class),
        )
        self.pre_layer = torch.nn.Dropout(0.7)

        self.atten = torch.nn.MultiheadAttention(self.hidden_size, self.head)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.attention:
            return self.hidden_size
        elif self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        h0 = torch.zeros(self.num_layers*2, batch['text'].size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, batch['text'].size(0), self.hidden_size).to(device)

        # Forward propagate
        feature = batch['text']
        feature = self.embed(feature) # [batch, max_len, emb_dim]
        feature = self.pre_layer(feature)

        
        feature = pack_padded_sequence(feature, batch['len'], batch_first=True)

        if self.model == 'lstm':
            out, (h, _) = self.lstm(feature, (h0, c0)) # out = [batch, seq_len, 2*hidden_size]
        elif self.model == 'gru':
            out, h = self.gru(feature, h0)
        elif self.model == 'rnn':
            out, h = self.rnn(feature, h0)
        else:
            raise ArgumentError
        
        # Decode the hidden state of the last time step
        if self.bidirectional: # h = [2 * layers, batch, hidden_size]
            if self.attention:
                h, _ = self.atten(h, h, h)
                h = torch.sum(h, dim=0)
            else:
                h = torch.cat((h[-1], h[-2]), dim=-1) # [batch, 2*hidden_size]
        else:
            h = h[-1]

        out = self.hidden2out(h) # [batch, num_class]

        return out
        raise NotImplementedError


class SlotTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        model: str,
        l1: int,
        pre_cnn
    ) -> None:
        super(SlotTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.pre_cnn = pre_cnn
        self.embedding_dim = embeddings.size(dim=1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.model = model
        self.l1 = l1
        
        self.rnn = torch.nn.RNN(
            input_size=self.encoder_input_size,
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout, 
            bidirectional=self.bidirectional,
            batch_first = True
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.encoder_input_size,
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout, 
            bidirectional=self.bidirectional,
            batch_first = True
        )
        self.gru = torch.nn.GRU(
            input_size=self.encoder_input_size,
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout, 
            bidirectional=self.bidirectional,
            batch_first = True
        )
        self.slot_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.7),
            torch.nn.Linear(self.encoder_output_size, self.l1),
            torch.nn.PReLU(),
        )
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, 1, 1),
            # torch.nn.BatchNorm1d(self.embedding_dim),
            # torch.nn.PReLU(),
        )
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv1d(self.embedding_dim, self.embedding_dim, 5, 1, 2),
            # torch.nn.BatchNorm1d(self.embedding_dim),
            # torch.nn.PReLU(),
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv1d(self.embedding_dim, self.embedding_dim, 7, 1, 3),
            # torch.nn.BatchNorm1d(self.embedding_dim),
            # torch.nn.PReLU(),
        )
        self.pre_layer = torch.nn.Sequential(
            torch.nn.Dropout(0.8)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size
        raise NotImplementedError
    
    @property
    def encoder_input_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.pre_cnn:
            return 3 * self.embedding_dim
        return self.embedding_dim

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        h0 = torch.zeros(self.num_layers*2, batch['tokens'].size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, batch['tokens'].size(0), self.hidden_size).to(device)

        # Forward propagate
        feature = batch['tokens']
        feature = self.embed(feature) # [batch, max_len, emb_dim]
        feature = self.pre_layer(feature)

        # CNN layer
        if self.pre_cnn:
            cnn_f = feature.permute(0, 2, 1) # [batch, emb_dim, max_len]
            cnn_f1 = self.cnn(cnn_f) 
            cnn_f2 = self.cnn1(cnn_f) 
            cnn_f3 = self.cnn2(cnn_f) 
            cnn_f1 = cnn_f1.permute(0, 2, 1)
            cnn_f2 = cnn_f2.permute(0, 2, 1)
            cnn_f3 = cnn_f3.permute(0, 2, 1)
            feature = torch.cat((cnn_f1, cnn_f2, cnn_f3), -1)

        # concat cnn
        # feature = torch.cat((feature, cnn_f), -1)

        # out = [batch, seq_len, 2*hidden_size]
        if self.model == 'lstm':
            out, (h, _) = self.lstm(feature, (h0, c0))
        elif self.model == 'gru':
            out, h = self.gru(feature, h0)
        elif self.model == 'rnn':
            out, h = self.rnn(feature, h0)
        else:
            raise ArgumentError

        out = self.slot_classifier(out) # [batch, seq_len, pred_class]

        return out.permute(0, 2, 1) # [batch, pred_class, seq_len]
        raise NotImplementedError
