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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from cs272_project.dataset.tsv_dataset import DFDataset
from cs272_project.load_model import load_model


def eval_model(checkpoint_dir, eval_tsv, block_size=512):
    model, tokenizer = load_model(checkpoint_dir)
    print(f"Vocab size = {tokenizer.vocab_size}")
    print(f"Reading from {eval_tsv}")

    tsv_df = pd.read_csv(eval_tsv, sep="\t", header=0, index_col=0)
    unique_queries = tsv_df["question"].unique()
    reciprocal_rank = []
    for qi, q in enumerate(unique_queries[:15]):
        print(f"query: {q}")
        query_candidates = tsv_df.loc[tsv_df["question"] == q].copy()

        dataset = DFDataset(tokenizer, query_candidates, batch_size=1, max_records=None)
        data_loader = DataLoader(dataset, batch_size=None)
        t = tqdm(data_loader, desc="Prediction")

        for step, (batch_lm, _) in enumerate(t):
            try:
                if not pd.isna(query_candidates.iloc[step]["mc_logits"]):
                    t.set_description("(SKIP)")
                    continue
            except KeyError:
                pass
            t.set_description("(EVAL)")
            mc_logits = model(batch_lm)[1]
            query_candidates.at[query_candidates.index[step], 'mc_logits'] = mc_logits[0, 1].item()
        temp = np.argsort(-query_candidates["mc_logits"])
        query_candidates.at[query_candidates.index[temp], "rank"] = np.arange(1, len(query_candidates) + 1)
        rank = query_candidates.loc[query_candidates['category'] == 4, "rank"].min()
        if not np.isnan(rank):
            reciprocal_rank.append(1 / rank)

        tsv_df.update(query_candidates)
    tsv_df.to_csv(eval_tsv, sep="\t", columns=["question", "context", "category", "mc_logits"])

    print(f"MRR: {np.mean(reciprocal_rank):.3f}")
    print(len( tsv_df.loc[~pd.isna(tsv_df["mc_logits"])]))
    query_candidates = tsv_df.loc[~pd.isna(tsv_df["mc_logits"])]
    y_true = (query_candidates['category'] == 4).astype(int).to_numpy()
    y_scores = query_candidates['mc_logits'].to_numpy()
    thres = np.linspace(0, 1, 20)
    precision = [precision_score(y_true, y_scores > tt, zero_division=1) for tt in thres]
    recall = [recall_score(y_true, y_scores > tt) for tt in thres]
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(5, 4))
    # plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f"AP={ap:.3f}", marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 0.02])
    plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(thres, precision)
    # plt.xlabel("Threshold")
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.grid()
    plt.tight_layout()
    output_fig = f"/home/tylee/sdb/nlp_workspace/plots/pr/pr_curve.png"
    plt.savefig(output_fig)
    plt.close()
    print(f"MAP: {ap:.3f}")
