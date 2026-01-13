# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:32:55 2025

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score



torch.manual_seed(args.seed)
unknown_threshold = args.unknown_threshold
max_bgc_length = args.max_bgc_length

training_data_file = open(args.data_set)
fp_file = open(args.fp_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token_list = []
token_list_file = open("token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt")
for line in token_list_file:
    token_list.append(line.replace("\n",""))

#load model from file
bgc_mlm_model = BGC_MLM_tools.BGC_MLM(
  vocab_size=len(token_list),
  seq_len=max_bgc_length,
  d_model=args.d_model,
  n_layers=args.n_layers,
  heads=args.heads,
  dropout=args.droupout
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda available")
print(torch.cuda.is_available())
mlm = BGC_MLM_tools.MLM(bgc_mlm_model, len(token_list))
mlm.load_state_dict(torch.load(args.model_name,weights_only=True, map_location=torch.device(device)))
mlm.to(device)

bgc_tokens = []
bgc_names = []

classification_types = []
fps = {}
for line in fp_file:
    split_line = line.replace("\n","").split(",")
    if "bgc_id" in line:
        for t in split_line[1:len(split_line)]:
        #for t in split_line[1:4]:
           if t not in classification_types:
               classification_types.append(t)
    else:
        bgc_name = split_line[0]
        if len(classification_types) == 0:
            #if no header then number classifications
            for i in range(1,len(split_line)):
                classification_types.append(i-1)    
        if bgc_name in fps:
            continue
        fps[bgc_name] = []
        i = 0
        for val in split_line[1:len(split_line)]:
        #for val in split_line[1:4]:
            fps[bgc_name].append(int(val))
            
i= 0
for line in training_data_file:
    filename = line.replace("\n","")
    if filename not in fps:
        continue
    bgc_name = filename
    infile = open(args.token_path + "/" + filename)
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
    bgc_names.append(bgc_name)           
    bgc_tokens.append(bgc_token)
    
    i += 1

print(str(len(bgc_tokens)) + " BGCs under length threshold")   

target_embedding = []
for name in bgc_names: 
    target_embedding.append(fps[name])
    
target_embedding_dataset = BGC_MLM_tools.BGCTargetEmbeddingDataset(
   bgc_tokens, token_list, target_embedding, seq_len=max_bgc_length)

train_set, val_set = torch.utils.data.random_split(target_embedding_dataset, [math.floor(args.train_fraction*len(bgc_names)), len(bgc_names) - math.floor(args.train_fraction*len(bgc_names))])

if device == "cpu":
    unmasked_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    unmasked_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")

#using pretrained model
target_embedding_model = BGC_MLM_tools.BGCMetricLearning(mlm.bgc_mlm,args.d_model,args.fp_size)
bert_trainer = BGC_MLM_tools.BGCMetricTrainer(target_embedding_model, unmasked_loader, val_set,device=device,loss_type=args.loss_type)   
epochs = args.epochs
target_embedding_model.to(device)

for epoch in range(epochs):
  print("starting epoch: " + str(epoch))
  bert_trainer.train(epoch)
  if epoch == 10:
      #torch.save(bert_lm.state_dict(), args.model_name + "_epoch20")
      loss_outfile = open("classification_loss_file.txt",'w')
      for i in range(0, len(bert_trainer.train_loss_list)):
          loss_outfile.write(str(i) + "," + str(bert_trainer.train_loss_list[i]) + "," + str(bert_trainer.val_loss_list[i]) + "\n")
      loss_outfile.close()
  if epoch == 20 or epoch == 50:
      #torch.save(bert_lm.state_dict(), args.model_name + "_epoch50")    
      loss_outfile = open("classification_loss_file.txt",'w')
      for i in range(0, len(bert_trainer.train_loss_list)):
          loss_outfile.write(str(i) + "," + str(bert_trainer.train_loss_list[i]) + "," + str(bert_trainer.val_loss_list[i]) + "\n")
      loss_outfile.close()
   
loss_outfile = open("classification_loss_file.txt",'w')
for i in range(0, len(bert_trainer.train_loss_list)):
    loss_outfile.write(str(i) + "," + str(bert_trainer.train_loss_list[i]) + "," + str(bert_trainer.val_loss_list[i]) + "\n")
loss_outfile.close()


torch.save(target_embedding_model.state_dict(), args.model_name + "_" +args.model_output)


