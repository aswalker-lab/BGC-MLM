# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 16:54:14 2025

@author: Allison Walker
"""
# adapted from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.compose import ColumnTransformer
import random
from pathlib import Path
import torch
import math
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import tqdm
from torch.utils.data import Dataset, DataLoader

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):
            # for each dimension of the each position
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe


class MLMEmbedding(torch.nn.Module):
    """
    MLM Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of MLMEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, sequence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self.token(sequence).to(device) + self.position(sequence).to(device)
        return self.dropout(x)


class MultiHeadedAttention(torch.nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(
            query.size(-1)
        )

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(context.shape[0], -1, self.heads * self.d_k)
        )

        # (batch_size, max_len, d_model)
        return self.output_linear(context)


class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model=768, heads=12, feed_forward_hidden=768 * 4, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(
            self.self_multihead(embeddings, embeddings, embeddings, mask)
        )
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class BGC_MLM(torch.nn.Module):
    """
    BGC-MLM model : Masked langauage model for BGCs
    """

    def __init__(
        self, vocab_size, seq_len=20, d_model=768, n_layers=12, heads=12, dropout=0.1
    ):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = MLMEmbedding(
            vocab_size=vocab_size, seq_len=seq_len, embed_size=d_model
        )

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [
                EncoderLayer(d_model, heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x


class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class MLM(torch.nn.Module):
    """
    Masked Language Model
    """

    def __init__(self, bgc_mlm: BGC_MLM, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bgc_mlm = bgc_mlm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.mask_lm = MaskedLanguageModel(self.bgc_mlm.d_model, vocab_size)
        self.mask_lm = torch.nn.DataParallel(self.mask_lm)
        self.mask_lm.to(device)

    def forward(self, x):
        x = self.bgc_mlm(x)
        return self.mask_lm(x)


class BGCMultiLabelClassifier(torch.nn.Module):
    def __init__(self, bgc_mlm, d_model, n_tasks, freeze=False):
        super(BGCMultiLabelClassifier, self).__init__()
        self.bgc_mlm = bgc_mlm
        if freeze:
            for param in bgc_mlm.parameters():
                param.requires_grad = False
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Linear(d_model, int(d_model)), torch.nn.ReLU()
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(int(d_model), n_tasks))
        self.d_model = d_model

    def forward(self, x):
        x = self.bgc_mlm(x)
        x = torch.mean(x, 1)
        x = self.dense_layers(x)
        x = self.output(x)
        return x


class BGCRegression(torch.nn.Module):
    def __init__(self, bgc_mlm, d_model, n_tasks, freeze=False):
        super(BGCRegression, self).__init__()
        self.bgc_mlm = bgc_mlm
        if freeze:
            for param in bgc_mlm.parameters():
                param.requires_grad = False
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Linear(d_model, int(d_model)), torch.nn.ReLU()
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(int(d_model), n_tasks))
        self.d_model = d_model

    def forward(self, x):
        x = self.bgc_mlm(x)
        x = torch.mean(x, 1)
        x = self.dense_layers(x)
        x = self.output(x)
        return x


class BGCMetricLearning(torch.nn.Module):
    def __init__(self, bgc_mlm, d_model, d_fp, freeze=False):
        super(BGCMetricLearning, self).__init__()
        self.bgc_mlm = bgc_mlm
        self.d_model = d_model
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Linear(d_model, int(d_model)), torch.nn.ReLU()
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(int(d_model), d_fp))

    def forward(self, x):
        x = self.bgc_mlm.embedding(x)
        x = torch.mean(x, 1)
        x = self.dense_layers(x)
        x = self.output(x)
        return x
