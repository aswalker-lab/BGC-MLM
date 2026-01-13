# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 19:41:55 2025

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
import tools.BGC_MLM_tools
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sys import argv
from src.utils import arg_parse

args = arg_parse.parse("classificationTask", argv[1:])
torch.manual_seed(args.seed)
unknown_threshold = args.unknown_threshold
max_bgc_length = args.max_bgc_length

test_data_file = open(args.data_set)
classification_file = open(args.classification_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token_list = []
token_list_file = open("token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt")
for line in token_list_file:
    token_list.append(line.replace("\n",""))

bgc_tokens = []
bgc_names = []

classification_types = []
classification_counts = {}
bgc_classifications = {}
total_count = 0
for line in classification_file:
    split_line = line.replace("\n","").split(",")
    if "bgc_id" in line:
        for t in split_line[1:len(split_line)]:
        #for t in split_line[1:4]:
           if t not in classification_types:
               classification_types.append(t)
               classification_counts[t] = 0
    else:
        bgc_name = split_line[0]
        if bgc_name in bgc_classifications:
            continue
        bgc_classifications[bgc_name] = []
        i = 0
        for val in split_line[1:len(split_line)]:
        #for val in split_line[1:4]:
            bgc_classifications[bgc_name].append(int(val))
            if int(val) == 1:
                classification_counts[classification_types[i]] += 1
                total_count += 1
            i += 1
            

i= 0
for line in test_data_file:
    filename = line.replace("\n","")
    if filename not in bgc_classifications:
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
    y_vals.append(bgc_classifications[name])
pos_weights = []
for i in range(0,len(y_vals[0])):
    pos_weights.append(1)

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

classification_model = BGC_MLM_tools.BGCMultiLabelClassifier(mlm.bgc_mlm,args.d_model,len(y_vals[0]),freeze=True)
classification_model.load_state_dict(torch.load(args.model_name,weights_only=True, map_location=torch.device(device)))
classification_model.to(device)

classification_dataset = BGC_MLM_tools.BGCClassificationDatasets(
   bgc_tokens, token_list, y_vals, seq_len=max_bgc_length)

if device == "cpu":
    unmasked_loader = DataLoader(
        classification_dataset, batch_size=1, shuffle=False, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    unmasked_loader = DataLoader(
        classification_dataset, batch_size=1, shuffle=False, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")
bert_trainer = BGC_MLM_tools.BGCMultiLabelTrainier(classification_model, unmasked_loader, classification_dataset, None, device=device)   
test_predictions = bert_trainer.predict(classification_dataset,batch_size=args.batch_size)

if not os.path.isfile( args.data_set[0:args.data_set.find(".")] + "_" + args.classification_file[0:args.classification_file.find(".")] + "_classification.txt"):
    outfile = open( args.data_set[0:args.data_set.find(".")] + "_" + args.classification_file[0:args.classification_file.find(".")] + "_classification.txt",'w')
    outfile.write("model_name,accuracy,balanced_accuracy,precision,recall,ROC_AUC,PRC_AUC\n")
else:
    outfile = open( args.data_set[0:args.data_set.find(".")] + "_" + args.classification_file[0:args.classification_file.find(".")] + "_classification.txt",'a')

for i in range(0, len(classification_types)):
    print(classification_types[i])
    
    #train
    data_iter = enumerate(unmasked_loader)
    true_y = []
    for j, data in data_iter:
        #print(data["classification_label"].shape)
        true_y.append(int(data["classification_label"].detach().cpu().numpy()[0,i]))
    if sum(true_y) < 2:
        print("not enough of class")
        continue
    print(sum(true_y))
    predicted_y = test_predictions.detach().cpu().numpy()[:,i]
    predicted_y_class = [x >0.5 for x in predicted_y]
    accuracy = accuracy_score(true_y, predicted_y_class)
    balanced_accuracy = balanced_accuracy_score(true_y, predicted_y_class)
    precision =precision_score(true_y, predicted_y_class)
    recall = recall_score(true_y, predicted_y_class)
    roc_auc = roc_auc_score(true_y, predicted_y)
    prc_auc = average_precision_score(true_y,predicted_y)
    outfile.write(args.model_name[args.model_name.rfind("/")+1:len(args.model_name)] + "_" + classification_types[i] + ",")
    outfile.write(str(accuracy) + "," + str(balanced_accuracy) + "," + str(precision) + "," + str(recall) + "," + str(roc_auc) + "," + str(prc_auc) + "\n")
    print("test accuracy: " +str(accuracy))
    print("test balanced accuracy: " +str(balanced_accuracy))
    print("test precision: " +str(precision))
    print("test recall: " +str(recall))
    print("test ROC AUC: " +str(roc_auc))
outfile.close()