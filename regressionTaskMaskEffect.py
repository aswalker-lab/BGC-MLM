# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 21:26:32 2026

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
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


torch.manual_seed(args.seed)
output_index = args.output_index

#read model parameters
for line in open(args.model_param_file):
    split_line = line.split(",")
    unknown_threshold = int(split_line[0])
    max_bgc_length  = int(split_line[1])
    d_model = int(split_line[2])
    n_layers = int(split_line[3])
    heads = int(split_line[4])
    droupout = float(split_line[5])
    batch_size = int(split_line[6])


test_data_file = open(args.data_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token_list = []
token_list_file = open("token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt")
for line in token_list_file:
    token_list.append(line.replace("\n",""))

bgc_tokens = []
bgc_names = []

regression_types = []


i= 0
for line in test_data_file:
    filename = line.replace("\n","")
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

y_vals = []
y = []
for i in range(0, args.num_tasks):
    y.append(0)
for name in bgc_names: 
    y_vals.append(y)
    
#load model from file
bgc_mlm_model = BGC_MLM_tools.BGC_MLM(
  vocab_size=len(token_list),
  seq_len=max_bgc_length,
  d_model=d_model,
  n_layers=n_layers,
  heads=heads,
  dropout=droupout
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda available")
print(torch.cuda.is_available())
mlm = BGC_MLM_tools.MLM(bgc_mlm_model, len(token_list))

regression_model = BGC_MLM_tools.BGCRegression(mlm.bgc_mlm,d_model,len(y_vals[0]),freeze=True)
regression_model.load_state_dict(torch.load(args.model_name,weights_only=True, map_location=torch.device(device)))
regression_model.to(device)

#do base predictions for each BGC
regression_dataset = BGC_MLM_tools.BGCRegressionDatasets(
   bgc_tokens, token_list, y_vals, seq_len=max_bgc_length)

if device == "cpu":
    unmasked_loader = DataLoader(
        regression_dataset, batch_size=1, shuffle=False, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    unmasked_loader = DataLoader(
        regression_dataset, batch_size=1, shuffle=False, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")
bert_trainer = BGC_MLM_tools.BGCRegressionTrainier(regression_model, unmasked_loader, regression_dataset, device=device)   
test_predictions = bert_trainer.predict(regression_dataset,batch_size=batch_size)
unmasked_prediction = test_predictions[0,output_index]
print(test_predictions[0,output_index])

#do masked predictions for each BGC
predictions = {}
for i in range(1,len(bgc_token)):
    token = bgc_tokens[0][i]
    if token == "PAD" or token == "SEP" or token == "CLS":
        continue
    masked_dataset = BGC_MLM_tools.specificMaskMLMDataset(
            bgc_tokens, token_list, [i], seq_len=max_bgc_length)
    if device == "cpu":
        masked_loader = DataLoader(
            masked_dataset, batch_size=1, shuffle=True, pin_memory=True)
    else:
        masked_loader = DataLoader(
            masked_dataset, batch_size=1, shuffle=True, pin_memory=False)
    bert_trainer = BGC_MLM_tools.BGCRegressionTrainier(regression_model, masked_loader, masked_dataset, device=device)   
    test_predictions = bert_trainer.predict(masked_dataset,batch_size=batch_size)
    
    diff = test_predictions[0,output_index] - unmasked_prediction
    predictions[str(i) + "_" + token] = diff

print(unmasked_prediction)
sorted_predictions = sorted(predictions.items(), key=lambda item: item[1])
print(sorted_predictions)
