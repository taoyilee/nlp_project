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
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

from transformers import GPT2PreTrainedModel, GPT2Model


class AS2HeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_neurons = 1024
        self.linear1 = nn.Linear(config.hidden_size, mlp_neurons)
        self.linear2 = nn.Linear(mlp_neurons, 1)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.ln_1 = nn.LayerNorm(mlp_neurons, eps=config.layer_norm_epsilon)

        self.act = F.relu
        self.act_out = torch.sigmoid

    def forward(self, hidden_states, cls_index=None):
        output = hidden_states[:, -1, :]
        output = self.linear1(output)
        output = self.act(output)
        output = self.ln_1(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.act_out(output)
        return output


class GPT2TANDAModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.as2_head = AS2HeadModel(config)

        self.init_weights()
        self.loss_fct = CrossEntropyLoss()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.as2_head(hidden_states)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss = self.loss_fct(mc_logits, mc_labels)
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()

            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)
