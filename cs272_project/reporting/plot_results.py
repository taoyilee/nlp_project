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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# https://gist.github.com/tomrunia/1e1d383fb21841e8f144

def plot_learning_curves(log_dir, plot_dir):
    log_name = Path(log_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)
    start_idx = 200
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())
    train_lm_loss = event_acc.Scalars('train_lm_loss')
    train_mc_loss = event_acc.Scalars('train_mc_loss')
    train_ppl = event_acc.Scalars('train_ppl')
    eval_perplexity = event_acc.Scalars('eval_perplexity')
    eval_lm_loss = event_acc.Scalars('eval_lm_loss')
    eval_mc_loss = event_acc.Scalars('eval_mc_loss')
    lr = event_acc.Scalars('lr')

    x = [t.step for t in train_lm_loss]
    y = [t.value for t in train_lm_loss]
    plt.plot(x[start_idx:], y[start_idx:], label="Train")
    x = [t.step for t in eval_lm_loss]
    y = [t.value for t in eval_lm_loss]
    plt.plot(x, y, label="Dev")
    plt.legend()
    plt.grid()
    # plt.title("Language Model Loss")
    plt.xlabel("Iteration #")
    plt.ylabel("Cross Entropy Loss")
    plt.ylim([1, 5])
    plt.tight_layout()
    plt.savefig(plot_dir / f"{log_name.name}_lm_loss.png")
    plt.savefig(plot_dir / f"{log_name.name}_lm_loss.pdf")

    plt.figure()
    fig, ax1 = plt.subplots()
    start_idx_as2 = 10
    x = [t.step for t in train_mc_loss]
    y = [t.value for t in train_mc_loss]
    plt.plot(x[start_idx_as2:], y[start_idx_as2:], label="Train", color="C0")
    x = [t.step for t in eval_mc_loss]
    y = [t.value for t in eval_mc_loss]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y, label="Dev", color="C1")
    ax1.set_ylabel('Train Cross Entropy Loss', color="C0")  # we already handled the x-label with ax1
    ax2.set_ylabel('Dev Cross Entropy Loss', color="C1")  # we already handled the x-label with ax1
    # ax1.legend()
    ax1.grid(True)
    # plt.title("AS2 Classification Head Loss")
    ax1.set_xlabel("Iteration #")
    # plt.ylabel("Cross Entropy Loss")
    # plt.ylim([0, 2])
    plt.tight_layout()
    plt.savefig(plot_dir / f"{log_name.name}_mc_loss.png")
    plt.savefig(plot_dir / f"{log_name.name}_mc_loss.pdf")
    plt.close()

    plt.figure()
    x = [t.step for t in train_ppl]
    y = [t.value for t in train_ppl]
    plt.plot(x[2:], y[2:], label="Train")
    x = [t.step for t in eval_perplexity]
    y = [t.value for t in eval_perplexity]
    plt.plot(x, y, label="Dev")
    plt.legend()
    plt.grid()
    plt.xlabel("Iteration #")
    plt.ylabel("Perplexity")
    plt.ylim([2, 20])
    plt.tight_layout()
    plt.savefig(plot_dir / f"{log_name.name}_perplexity.png")
    plt.savefig(plot_dir / f"{log_name.name}_perplexity.pdf")

    plt.close()
    plt.figure()
    ratio = 1

    x = np.array([t.step for t in train_lm_loss]) + np.array([t.step for t in train_mc_loss])
    y = np.array([t.value for t in train_lm_loss]) * ratio + np.array([t.value for t in train_mc_loss])

    plt.plot(x[start_idx:], y[start_idx:], label="Train")
    x = np.array([t.step for t in eval_lm_loss]) + np.array([t.step for t in eval_mc_loss])
    y = np.array([t.value for t in eval_lm_loss]) * ratio + np.array([t.value for t in eval_mc_loss])
    y_min = np.argmin(y)
    plt.plot(x[2:], y[2:], label="Dev")
    plt.scatter(x[y_min], y[y_min], label="Dev (min)", marker="x")
    plt.text(x[y_min], y[y_min], f"Iter {x[y_min]}")
    plt.legend()
    # plt.ylim([0, 5])
    plt.grid()
    # plt.title("Combined Loss")
    plt.xlabel("Iteration #")
    plt.ylabel("Cross Entropy Loss")
    plt.tight_layout()
    plt.savefig(Path(plot_dir) / f"{log_name.name}_tot_loss.png")
    plt.savefig(Path(plot_dir) / f"{log_name.name}_tot_loss.pdf")
