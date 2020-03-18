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

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from cs272_project.dataset.tsv_dataset import DFDataset
from cs272_project.load_model import load_model


def eval_model(checkpoint_dir, eval_tsv, block_size=512):
    model, tokenizer = load_model(checkpoint_dir)
    print(f"Vocab size = {tokenizer.vocab_size}")
    print(f"Reading from {eval_tsv}")

    tsv_df = pd.read_csv(eval_tsv, sep="\t", header=0, index_col=0)
    dataset = DFDataset(tokenizer, tsv_file=eval_tsv, batch_size=1, max_records=None)
    data_loader = DataLoader(dataset, batch_size=None)
    t = tqdm(data_loader, desc="Prediction")
    i = 0
    j = 0
    max_samples = 10000
    for step, (batch_lm, _) in enumerate(t):
        try:
            if not pd.isna(tsv_df.iloc[step]["mc_logits"]):
                t.set_description("(SKIP)")
                j += 1
                continue
        except KeyError:
            pass

        if i + j < max_samples or tsv_df.at[step, "category"] == 4:
            t.set_description("(EVAL)")
            i += 1
            mc_logits = model(batch_lm)[1]
            tsv_df.at[step, "mc_logits"] = mc_logits[0, 1].item()
        else:
            t.set_description("(SKIP)")
        if i % 1000 == 0:
            tsv_df.to_csv(eval_tsv, sep="\t", columns=["question", "context", "category", "mc_logits"])
    tsv_df.to_csv(eval_tsv, sep="\t", columns=["question", "context", "category", "mc_logits"])
