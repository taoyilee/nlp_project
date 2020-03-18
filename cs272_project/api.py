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


def fine_tune(config, model="gpt2", batch_size=4, train_tsv=None, dev_tsv=None):
    from cs272_project.fine_tuning import main
    output_dir = Path(config.default.output_dir)
    existing_dirs = glob(str(output_dir / "exp_*"))

    if not existing_dirs:
        last_index = 0
    else:
        existing_dirs.sort()
        last_index = int(Path(existing_dirs[-1]).name.replace("exp_", ""))

    output_dir = output_dir / f"exp_{last_index + 1:05d}"
    print(f'writing output to {output_dir}')
    if train_tsv is not None and dev_tsv is not None:
        main(["--output_dir", str(output_dir),
              "--model_type", model,
              "--train_tsv", train_tsv,
              "--dev_tsv", dev_tsv,
              "--num_train_epochs", "4",
              "--model_name_or_path", model,
              "--per_gpu_train_batch_size", f"{batch_size}",
              "--per_gpu_eval_batch_size", f"{batch_size}"])
    else:
        main(["--output_dir", config.default.output_dir,
              "--model_type", model,
              "--num_train_epochs", "4",
              "--model_name_or_path", model,
              "--per_gpu_train_batch_size", f"{batch_size}",
              "--per_gpu_eval_batch_size", f"{batch_size}"])
