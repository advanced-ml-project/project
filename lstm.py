'''
lstm.py

a pyTorch model object

based on Raymond Cheng
https://github.com/itsuncheng/fake_news_classification

& 

PPHA 30255 homework 4

'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):

    def __init__(self, vocab_len, dropout=0.5, dimension=100, layers=1):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_len, dimension*3)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=dimension*3,
                            hidden_size=dimension,
                            num_layers=layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=dropout)

        self.fc = nn.Linear(2*dimension, 3)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, 
                                            batch_first=True, 
                                            enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.softmax(text_fea, dim=1)
        return text_out

