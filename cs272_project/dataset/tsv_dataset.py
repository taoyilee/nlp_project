#  MIT License
#  #
#  Copyright (c) 2020, Michael Tao-Yi Lee
#  #
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  #
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TSVDataset(Dataset):
    _len = None

    def __init__(self, tokenizer: PreTrainedTokenizer, tsv_file,
                 batch_size=32, block_size=512):
        print(f"Creating features from tsv_file {tsv_file}")

        self.tokenizer = tokenizer
        self._df = pd.read_csv(tsv_file, delimiter="\t", dtype={"category": int}, index_col=0)
        self.block_size = block_size
        self._ohe = OneHotEncoder(sparse=False)
        self._ohe.fit(self._df["category"].to_numpy().reshape(-1, 1))
        self.batch_size = batch_size

    def __len__(self):
        if self._len is None:
            self._len = math.ceil(len(self._df) / self.batch_size)
        return self._len

    def __getitem__(self, i):
        rows = self._df.iloc[i * self.batch_size:(i + 1) * self.batch_size]

        def format_string(r):
            return f"{r[1]['question']} [SEP] {r[1]['context']} [CLS]"

        def tokenize(r):
            return torch.tensor(self.tokenizer.encode(r, max_length=self.block_size), dtype=torch.long)

        def collate(r):
            return pad_sequence(r, batch_first=True)

        inputs = list(map(format_string, rows.iterrows()))
        inputs = list(map(tokenize, inputs))
        # label_tensor = torch.tensor(self._ohe.transform(rows['category'].to_numpy().reshape(-1, 1)), dtype=torch.long)
        label_tensor = torch.tensor(rows['category'].to_numpy() - 1, dtype=torch.long)
        return collate(inputs), label_tensor
