# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:11:39 2025

@author: Allison Walker
"""
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.compose import ColumnTransformer
import random
from pathlib import Path
import torch
import math
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
import tools.BGC_MLM_tools
from src.utils import arg_parse
from sys import argv

args = arg_parse.parse_args("top_k_prediction", argv[1:])
unknown_threshold = args.unknown_threshold
max_bgc_length = args.max_bgc_length

# TODO: make in dir an argument and fix train.txt to have shorter path
training_data_file = open(args.data_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load token list
token_list = []
token_list_file = open("token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt")
for line in token_list_file:
    token_list.append(line.replace("\n",""))

# load model from file
bgc_mlm_model = BGC_MLM_tools.BGC_MLM(
  vocab_size=len(token_list),
  seq_len=max_bgc_length,
  d_model=args.d_model,
  n_layers=args.n_layers,
  heads=args.heads,
  dropout=args.dropout
)

mlm = BGC_MLM_tools.MLM(bgc_mlm_model, len(token_list))
mlm.load_state_dict(torch.load(args.model_name,weights_only=True, map_location=torch.device(device)))
mlm.to(device)

i= 0
bgc_tokens = []
bgc_names = []
count = 0
bgc_token_dic = {}
for line in training_data_file:
    filename = line.replace("\n","")
    infile = open(line.replace("\n",""))
    bgc_token = ["CLS"]
    useBGC = True
    for line in infile:
        split_line = line.split(",")
    if len(split_line) + 2 > max_bgc_length:
        continue
    for t in split_line:
        if t in token_list:
            bgc_token.append(t)
        else:
            bgc_token.append("UNK")
    bgc_token.append("SEP")
    if len(bgc_token) < max_bgc_length:
        j = len(bgc_token)
        while j < max_bgc_length:
           bgc_token.append("PAD")
           j += 1
    bgc_tokens.append(bgc_token)
    i += 1

print(str(len(bgc_tokens)) + " BGCs under length threshold")
outfile = open(args.model_name + "_topK_accuracy.csv",'w')
outfile.write("avg loss,avg accuracy, avg top 5, top 10\n")
avg_loss = 0
avg_accuracy= 0
avg_5_accuracy = 0
avg_10_accuracy = 0
max_bgc_length = 0

accuracy_by_position = []
max_val_by_position = []
for i in range(1,len(bgc_token)):
    masked_dataset = BGC_MLM_tools.specificMaskMLMDataset(
            bgc_tokens, token_list, [i], seq_len=max_bgc_length)
    if device == "cpu":
        masked_loader = DataLoader(
            masked_dataset, batch_size=1, shuffle=True, pin_memory=True)
    else:
        masked_loader = DataLoader(
            masked_dataset, batch_size=1, shuffle=True, pin_memory=False)
    mlm_trainer = BGC_MLM_tools.MLMTrainer(mlm, masked_loader, masked_loader, device=device)   
    pos_avg_loss, pos_avg_accuracy, pos_avg_5_accuracy, pos_avg_10_accuracy, avg_max_val = mlm_trainer.predictSequenceMetrics()
    if pos_avg_loss == None:
        continue
    max_bgc_length += 1
    avg_loss += pos_avg_loss
    avg_accuracy += pos_avg_accuracy
    avg_5_accuracy += pos_avg_5_accuracy
    avg_10_accuracy += pos_avg_10_accuracy
    accuracy_by_position.append(pos_avg_accuracy)
    max_val_by_position.append(avg_max_val)
avg_loss /= max_bgc_length
avg_accuracy /= max_bgc_length
avg_5_accuracy /= max_bgc_length
avg_10_accuracy /= max_bgc_length

outfile.write(str(avg_loss) + "," + str(avg_accuracy) + ","+ str(avg_5_accuracy) + ","+ str(avg_10_accuracy) +"\n")
outfile.close()

outfile = open(args.model_name + "_accuracy_by_position.csv",'w')
for i in range(0, len(accuracy_by_position)):
    outfile.write(str(i) + "," + str(accuracy_by_position[i]) + "," + str(max_val_by_position[i]) + "\n")
outfile.close()
