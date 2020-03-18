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

import torch
from transformers import GPT2Config, GPT2Tokenizer

from cs272_project.model import GPT2TANDAModel


def load_model(checkpoint_dir):
    ckpt_dir = Path(checkpoint_dir)
    config = GPT2Config.from_pretrained(ckpt_dir / "config.json")
    tokenizer = GPT2Tokenizer.from_pretrained(str(ckpt_dir))
    model = GPT2TANDAModel(config)

    model.load_state_dict(torch.load(ckpt_dir / "pytorch_model.bin"))
    model.eval()
    return model, tokenizer
