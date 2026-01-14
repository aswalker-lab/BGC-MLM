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

class BGCClassificationDatasets(Dataset):
    """Dataset for classification task"""

    def __init__(self, data, token_list, classifications, seq_len=20):
        # self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.classifications = classifications

    def __len__(self):
        return self.corpus_lines

    def get_sent(self, index):
        """return random sentence pair"""
        return self.lines[index]

    def get_classification(self, index):
        return self.classifications[index]

    def makeItem(self, sequence, token_list, classification_label):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            # don't change SEP or CLS
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue

            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {
            "bert_input": output,
            "bert_label": output_label,
            "classification_label": classification_label,
        }
        # return {key: torch.tensor(value).cuda() for key, value in output.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}

    def __getitem__(self, item):
        bgc = self.get_sent(item)
        classification = self.get_classification(item)
        bgc_object = self.makeItem(bgc, self.token_list, classification)
        # Step 2: replace random words in sentence with mask / random words
        return bgc_object


class BGCRegressionDatasets(Dataset):
    """Dataset for classification task"""

    def __init__(self, data, token_list, values, seq_len=20):
        # self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.values = values

    def __len__(self):
        return self.corpus_lines

    def get_sent(self, index):
        """return random sentence pair"""
        return self.lines[index]

    def get_values(self, index):
        return self.values[index]

    def makeItem(self, sequence, token_list, regression_value):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            # don't change SEP or CLS
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue

            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {
            "bert_input": output,
            "bert_label": output_label,
            "regression_value": regression_value,
        }
        # return {key: torch.tensor(value).cuda() for key, value in output.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}

    def __getitem__(self, item):
        bgc = self.get_sent(item)
        values = self.get_values(item)
        bgc_object = self.makeItem(bgc, self.token_list, values)
        # Step 2: replace random words in sentence with mask / random words
        return bgc_object


class BGCTargetEmbeddingDataset(Dataset):
    """Dataset for target embedding task"""

    def __init__(self, data, token_list, target_embedding, seq_len=20):
        # self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.target_embedding = target_embedding

    def __len__(self):
        return self.corpus_lines

    def get_sent(self, index):
        """return random sentence pair"""
        return self.lines[index]

    def get_target_embedding(self, index):
        return self.target_embedding[index]

    def makeItem(self, sequence, token_list, target_embedding):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            # don't change SEP or CLS
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue

            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {
            "bert_input": output,
            "bert_label": output_label,
            "target_embedding": target_embedding,
        }
        # return {key: torch.tensor(value).cuda() for key, value in output.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}

    def __getitem__(self, item):
        bgc = self.get_sent(item)
        target_embedding = self.get_target_embedding(item)
        bgc_object = self.makeItem(bgc, self.token_list, target_embedding)
        # Step 2: replace random words in sentence with mask / random words
        return bgc_object
