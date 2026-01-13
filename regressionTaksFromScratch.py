# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:35:52 2025

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
unknown_threshold = args.unknown_threshold
max_bgc_length = args.max_bgc_length

training_data_file = open(args.data_set)
regression_file = open(args.regression_file)
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

mlm.to(device)

bgc_tokens = []
bgc_names = []

regression_types = []
bgc_values = {}
for line in regression_file:
    split_line = line.replace("\n","").split(",")
    if "bgc_id" in line:
        for t in split_line[1:len(split_line)]:
        #for t in split_line[1:4]:
           if t not in regression_types:
               regression_types.append(t)
    else:
        bgc_name = split_line[0]
        if bgc_name in bgc_values:
            continue
        bgc_values[bgc_name] = []
        i = 0
        for val in split_line[1:len(split_line)]:
        #for val in split_line[1:4]:
            bgc_values[bgc_name].append(float(val))
            
i= 0
for line in training_data_file:
    filename = line.replace("\n","")
    if filename not in bgc_values:
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
y_vals = []
for name in bgc_names: 
    y_vals.append(bgc_values[name])

regression_dataset = BGC_MLM_tools.BGCRegressionDatasets(
   bgc_tokens, token_list, y_vals, seq_len=max_bgc_length)

train_set, val_set = torch.utils.data.random_split(regression_dataset, [math.floor(9*len(bgc_names)/10), len(bgc_names) - math.floor(9*len(bgc_names)/10)])

if device == "cpu":
    unmasked_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    unmasked_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")

#using pretrained model

regression_model = BGC_MLM_tools.BGCRegression(mlm.bgc_mlm,args.d_model,len(y_vals[0]))
bert_trainer = BGC_MLM_tools.BGCRegressionTrainier(regression_model, unmasked_loader, val_set, device=device)   
epochs = 400
regression_model.to(device)

for epoch in range(epochs):
  print("starting epoch: " + str(epoch))
  bert_trainer.train(epoch)
  if epoch == 10:
      #torch.save(bert_lm.state_dict(), args.model_name + "_epoch20")
      loss_outfile = open("regression_loss_file.txt",'w')
      for i in range(0, len(bert_trainer.train_loss_list)):
          loss_outfile.write(str(i) + "," + str(bert_trainer.train_loss_list[i]) + "," + str(bert_trainer.val_loss_list[i]) + "\n")
      loss_outfile.close()
  if epoch == 20 or epoch == 50:
      #torch.save(bert_lm.state_dict(), args.model_name + "_epoch50")    
      loss_outfile = open("regression_loss_file.txt",'w')
      for i in range(0, len(bert_trainer.train_loss_list)):
          loss_outfile.write(str(i) + "," + str(bert_trainer.train_loss_list[i]) + "," + str(bert_trainer.val_loss_list[i]) + "\n")
      loss_outfile.close()
   
loss_outfile = open("regression_loss_file.txt",'w')
for i in range(0, len(bert_trainer.train_loss_list)):
    loss_outfile.write(str(i) + "," + str(bert_trainer.train_loss_list[i]) + "," + str(bert_trainer.val_loss_list[i]) + "\n")
loss_outfile.close()

#check metrics
val_predictions = bert_trainer.predict(val_set,batch_size=args.batch_size)
train_predictions = bert_trainer.predict(train_set,batch_size=args.batch_size)
val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, pin_memory=False)
test_loader = DataLoader(
        train_set, batch_size=1, shuffle=False, pin_memory=False)
print(val_predictions.shape)
for i in range(0, len(regression_types)):
    print(regression_types[i])
    
    #train
    data_iter = enumerate(test_loader)
    true_y = []
    for j, data in data_iter:
        true_y.append(int(data["regression_value"].detach().cpu().numpy()[0,i]))
    predicted_y = train_predictions.detach().cpu().numpy()[:,i]
    mse = mean_squared_error(true_y,predicted_y)
    mae = mean_absolute_error(true_y,predicted_y)
    min_train = min(min(true_y),min(predicted_y))
    max_train = max(max(true_y),max(predicted_y))
    corr = pearsonr(true_y,predicted_y)
    print("Train MSE: " + str(mse))
    print("Train MAE: " + str(mae))
    print("Pearson's correlation: " + str(corr))
    plt.scatter(true_y, predicted_y,c='blue')
    #val
    data_iter = enumerate(val_loader)
    true_y = []
    for j, data in data_iter:
        #print(data["classification_label"].shape)
        true_y.append(int(data["regression_value"].detach().cpu().numpy()[0,i]))
    predicted_y = val_predictions.detach().cpu().numpy()[:,i]
    mse = mean_squared_error(true_y,predicted_y)
    mae = mean_absolute_error(true_y,predicted_y)
    corr = pearsonr(true_y,predicted_y)
    print("Val MSE: " + str(mse))
    print("Val MAE: " + str(mae))
    print("Pearson's correlation: " + str(corr))
    plt.scatter(true_y, predicted_y,c='orange')
    plt.xlim(min(0,min(true_y),min(predicted_y),min_train),max(max(true_y),max(predicted_y),max_train))
    plt.ylim(min(0,min(true_y),min(predicted_y),min_train),max(max(true_y),max(predicted_y),max_train))
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title(regression_types[i])
    plt.savefig(regression_types[i] + ".png")
    plt.cla()

torch.save(regression_model.state_dict(), args.model_name)