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

from glob import glob
from pathlib import Path

import pandas as pd


def asnq_subsampler(dataset_dir, outdir, samples_each_category=500):
    dataset = Path(dataset_dir)
    if not dataset.is_dir():
        raise FileNotFoundError(f"{dataset} does not exist")
    print(f"Reading from {dataset_dir}")
    output = Path(outdir)
    output.mkdir(exist_ok=True)
    tsv_files = glob(str(dataset / "*.tsv"))
    tsv_files.sort()
    categories = {i + 1: [] for i in range(4)}
    for t in tsv_files:
        tsv_df = pd.read_csv(t, delimiter="\t", names=['question', 'context', 'category'])
        for i in range(4):
            categories[i + 1].extend(tsv_df.loc[tsv_df['category'] == i + 1].to_dict(orient="records"))
        if all([len(v) > samples_each_category for v in categories.values()]):
            break
    categories = {k: v[:samples_each_category] for k, v in categories.items()}
    df = pd.DataFrame(sum(categories.values(), []))
    output_file = output / "subsampled_asnq.tsv"
    df.to_csv(output_file, sep="\t")
    return output_file


def dataset_split(tsv_file, training_ratio=1.0, dev_ratio=0.0):
    tsv_df = pd.read_csv(tsv_file, delimiter="\t", names=['question', 'context', 'category'], index_col=0, header=0)
    training_samples = int(len(tsv_df) * training_ratio)
    development_samples = int(len(tsv_df) * dev_ratio)
    tsv_df = tsv_df.sample(frac=1)
    train_df = tsv_df.iloc[:training_samples].reset_index(drop=True)
    dev_df = tsv_df.iloc[training_samples:training_samples + development_samples].reset_index(drop=True)
    test_df = tsv_df.iloc[training_samples + development_samples:].reset_index(drop=True)
    print(f"Train: {len(train_df)} {100 * len(train_df) / len(tsv_df):.2f}%")
    print(f"Dev: {len(dev_df)} {100 * len(dev_df) / len(tsv_df):.2f}%")
    print(f"Test: {len(test_df)} {100 * len(test_df) / len(tsv_df):.2f}%")

    train_df.to_csv(tsv_file.parent / "train.tsv", sep="\t")
    if training_ratio < 1.0:
        dev_df.to_csv(tsv_file.parent / "dev.tsv", sep="\t")
        test_df.to_csv(tsv_file.parent / "test.tsv", sep="\t")
