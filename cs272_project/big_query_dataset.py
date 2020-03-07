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

import torch
from google.cloud import bigquery
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BigQueryDataset(Dataset):
    _len = None

    def __init__(self, tokenizer: PreTrainedTokenizer, project_name="focus-empire-270208", table_name="asnq.train",
                 batch_size=32, block_size=512):
        print(f"Creating features from table {project_name}.{table_name}")

        self.tokenizer = tokenizer
        self.project_name = project_name
        self.table_name = table_name
        self.client = bigquery.Client(project=self.project_name)
        self.block_size = block_size
        self.batch_size = batch_size

    def __len__(self):
        if self._len is None:
            QUERY = ('SELECT '
                     'COUNT(*) as total_rows '
                     f'FROM `{self.table_name}`')
            query_job = self.client.query(QUERY)
            rows = query_job.result().to_dataframe()
            self._len = math.ceil(rows.loc[0, "total_rows"] / self.batch_size)
        return self._len

    def __getitem__(self, i):
        QUERY = ('SELECT * '
                 f'FROM `{self.table_name}` '
                 f'LIMIT {self.batch_size} OFFSET {i * self.batch_size}')
        query_job = self.client.query(QUERY)

        rows = query_job.result().to_dataframe()

        def format_string(r):
            return f"{r[1]['question']} [SEP] {r[1]['context']} [CLS]"

        def tokenize(r):
            return torch.tensor(self.tokenizer.encode(r, max_length=self.block_size), dtype=torch.long)

        def collate(r):
            return pad_sequence(r, batch_first=True)

        inputs = list(map(format_string, rows.iterrows()))
        inputs = list(map(tokenize, inputs))

        return collate(inputs), torch.tensor(rows['label'])
