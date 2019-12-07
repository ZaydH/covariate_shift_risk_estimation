import logging
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from ._utils import TORCH_DEVICE


class ClassifierConfig:
    LOGGER_NAME = 'nlp_learner'
    LOG_LEVEL = logging.DEBUG

    NUM_EPOCH = 50
    BATCH_SIZE = None

    LEARNING_RATE = 1E-3
    # WEIGHT_DECAY = 5E-3
    # LEARNING_RATE = 1E-4
    WEIGHT_DECAY = 1E-4

    # Based on the PUbN paper.
    # 9216 = 3 Layers * 3 (Min/Max/Avg) * 1024
    PREPROCESS_DIM = 9216

    BIDIRECTIONAL = True
    EMBED_DIM = 300

    RNN_HIDDEN_DIM = 300
    RNN_DEPTH = 1

    FF_HIDDEN_DEPTH = 2
    FF_HIDDEN_DIM = 300
    FF_ACTIVATION = nn.ReLU

    BASE_RNN = nn.LSTM


class BaseClassifier(nn.Module):
    Config = ClassifierConfig

    def __init__(self, embed: Optional[Tensor]):
        super().__init__()
        if embed is None:
            self._embed = self._rnn = None
            in_dim = self.Config.PREPROCESS_DIM
        else:
            self._embed = nn.Embedding.from_pretrained(embed.clone(), freeze=False)

            self._rnn = self.Config.BASE_RNN(num_layers=self.Config.RNN_DEPTH,
                                             hidden_size=self.Config.RNN_HIDDEN_DIM,
                                             input_size=self.Config.EMBED_DIM,
                                             bidirectional=self.Config.BIDIRECTIONAL)
            in_dim = self.Config.RNN_HIDDEN_DIM << (1 if self.Config.BIDIRECTIONAL else 0)

        self._ff = nn.Sequential()
        for i in range(1, self.Config.FF_HIDDEN_DEPTH + 1):
            self._ff.add_module(f"Hidden_{i:02}_Lin", nn.Linear(in_dim, self.Config.FF_HIDDEN_DIM))
            self._ff.add_module(f"Hidden_{i:02}_Act", self.Config.FF_ACTIVATION())
            # self._ff.add_module("Hidden_{i:02}_BN", nn.BatchNorm1d(self.Config.FF_HIDDEN_DIM))
            in_dim = self.Config.FF_HIDDEN_DIM
        # Add output layer
        self._ff.add_module("FF_Out", nn.Linear(in_dim, 1))
        # self._ff.add_module("FF_Out_Act", nn.Sigmoid())

        self.to(TORCH_DEVICE)

    def is_rnn(self) -> bool:
        r""" Returns \p True if the object has an RNN"""
        return self._rnn is not None

    # noinspection PyUnresolvedReferences
    def forward(self, x: Tensor, seq_len: Optional[Tensor] = None) -> Tensor:
        if self.is_rnn():
            return self._forward_rnn(x, seq_len)
        assert seq_len is None, "Sequence length not valid for FF RNN"
        assert x.shape[-1] == self.Config.PREPROCESS_DIM, "Unknown FF dimension"
        return self._forward_ff(x)

    def _forward_ff(self, x: Tensor) -> Tensor:
        r""" Forward method if the block is only a FF block"""
        assert len(x.shape) == 2, "FF base classifier can only have two dimensions"
        return self._ff.forward(x).squeeze(dim=1)

    def _forward_rnn(self, x: Tensor, seq_len: Optional[Tensor]) -> Tensor:
        r""" Forward method when the base classifier is an RNN"""
        assert self.is_rnn(), "Cannot call forward_rnn method if not an actual RNN"
        assert seq_len is not None, "Sequence length required if doing an RNN test"

        batch_size = x.shape[1]
        assert batch_size == seq_len.numel(), "Number of elements mismatch"
        x_embed = self._embed(x)

        # **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
        seq_out, _ = self._rnn.forward(x_embed, hx=None)  # Always use a fresh hidden

        # Need to subtract 1 since to correct length for base
        ff_in = seq_out[seq_len - 1, torch.arange(batch_size)]  # ToDo verify correctness
        assert ff_in.shape[0] == batch_size, "Batch size mismatch"
        return self._forward_ff(ff_in)
