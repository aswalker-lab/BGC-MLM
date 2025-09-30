# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 09:35:51 2025

@author: Allison Walker
"""

#adapted from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

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
import BGC_MLM_tools
torch.manual_seed(0)

#TODO: try type of position encoder used by ESM?
#TODO: try using ESM PFAM encondings

parser = argparse.ArgumentParser()            
parser.add_argument('model_name')           # positional argument
parser.add_argument('unknown_threshold',type=int)           # positional argument
parser.add_argument('max_bgc_length',type=int)           # positional argument
parser.add_argument('d_model',type=int)           # embedding size
parser.add_argument('n_layers',type=int)           # positional argument
parser.add_argument('heads',type=int)           # positional argument
parser.add_argument('dropout',type=float)           # positional argument
parser.add_argument('batch_size',type=int) #batch size

args = parser.parse_args()

print(args.model_name + "\n")
print("unknown threshold: " + str(args.unknown_threshold) + "\n")
print("max BGC length: " + str(args.max_bgc_length) + "\n")
print("Embedding Size: " + str(args.d_model) + "\n")
print("Number of layers: " + str(args.n_layers) + "\n")
print("Number of attention heads: " + str(args.heads) + "\n")
print("Dropout: " + str(args.dropout) + "\n")
print("Batch size: " + str(args.batch_size) + "\n")
torch.set_num_threads(8)
tokenized_dir = "tokenized_bgcs/pfam/"
pad_token = "PAD"
token_list  = ["PAD","CLS","SEP","MASK","UNK"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unknown_threshold = args.unknown_threshold
max_bgc_length = args.max_bgc_length

training_data_file = open("train.txt")
bgc_tokens = []
token_counts = {}
token_counts["CLS"] = 0
token_counts["SEP"] = 0
i = 0
print("loading tokens")
for line in training_data_file:
    #if i > 100000:
    #    break
    infile = open(line.replace("\n",""))
    bgc_token = ["CLS"]
    useBGC = True
    for line in infile:
        split_line = line.split(",")
        if len(split_line) + 2 > max_bgc_length:
            useBGC = False
            break
        for t in split_line:
            bgc_token.append(t)
            if t not in token_counts:
                token_counts[t] = 0
            token_counts[t] += 1
    if not useBGC:
        continue
    bgc_token.append("SEP")
    token_counts["CLS"] += 1
    token_counts["SEP"] += 1
    bgc_tokens.append(bgc_token)
    i += 1

print(str(len(bgc_tokens)) + " BGCs under length threshold")            
#update token list and replace unknowns
new_tokens = []
for tokens in bgc_tokens:
    new_bgc_token = []
    for t in tokens:
        if token_counts[t] < unknown_threshold:
            new_bgc_token.append("UNK")
        else:
            new_bgc_token.append(t)
            if t not in token_list:
                token_list.append(t)
    new_tokens.append(new_bgc_token)
bgc_tokens = new_tokens
print(str(len(token_list)) + " tokens past count threshold")

outfile = open("token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt",'w')
for t in token_list:
    outfile.write(t + "\n")
outfile.close()

#Pad bgcs
padded_bgcs = []
for bgc in bgc_tokens:
   i = 0
   if len(bgc) < max_bgc_length:
       j = len(bgc)
       while j < max_bgc_length:
           bgc.append("PAD")
           j += 1
   padded_bgcs.append(bgc)
   
dataset = BGC_MLM_tools.MLMDataset(
   padded_bgcs, token_list, seq_len=max_bgc_length)

train_set, val_set = torch.utils.data.random_split(dataset, [math.floor(9*len(padded_bgcs)/10), len(padded_bgcs) - math.floor(9*len(padded_bgcs)/10)])

print("train len " + str(len(train_set)))
print("val len " + str(len(val_set)))
print("vocab size " + str(len(token_list)))
print(len(bgc_tokens))
print(padded_bgcs[0])
bgc_mlm_model = BGC_MLM_tools.BGC_MLM(
  vocab_size=len(token_list),
  seq_len=max_bgc_length,
  d_model=args.d_model,
  n_layers=args.n_layers,
  heads=args.heads,
  dropout=args.dropout
)

mlm = BGC_MLM_tools.MLM(bgc_mlm_model, len(token_list))
mlm.to(device)

for name, param in mlm.named_parameters():
    print(name)
    print(param.device)

for name, buffer in mlm.named_buffers():
    print(name)
    print(buffer.device)

train_loader = DataLoader(
   train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")

mlm_trainer = BGC_MLM_tools.MLMTrainer(mlm, train_loader, val_set, device=device)   
epochs = 50

for epoch in range(epochs):
  print("starting epoch: " + str(epoch))
  mlm_trainer.train(epoch)
  if epoch == 10:
      loss_outfile = open("loss_file_" + args.model_name + ".txt",'w')
      for i in range(0, len(mlm_trainer.train_loss_list)):
          loss_outfile.write(str(i) + "," + str(mlm_trainer.train_loss_list[i]) + "," + str(mlm_trainer.val_loss_list[i]) + "\n")
      loss_outfile.close()
  if epoch == 25:
      loss_outfile = open("loss_file_" + args.model_name + ".txt",'w')
      for i in range(0, len(mlm_trainer.train_loss_list)):
          loss_outfile.write(str(i) + "," + str(mlm_trainer.train_loss_list[i]) + "," + str(mlm_trainer.val_loss_list[i]) + "\n")
      loss_outfile.close()
   
loss_outfile = open("loss_file_" + args.model_name + ".txt",'w')
for i in range(0, len(mlm_trainer.train_loss_list)):
    loss_outfile.write(str(i) + "," + str(mlm_trainer.train_loss_list[i]) + "," + str(mlm_trainer.val_loss_list[i]) + "\n")
loss_outfile.close()
torch.save(mlm.state_dict(), args.model_name)

