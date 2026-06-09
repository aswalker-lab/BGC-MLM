# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 14:56:36 2025

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
import BGC_MLM_tools
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score


parser = argparse.ArgumentParser()            
parser.add_argument('model_name')           # positional argument
parser.add_argument('unknown_threshold',type=int)           # positional argument
parser.add_argument('max_bgc_length',type=int)           # positional argument
parser.add_argument('d_model',type=int)           # positional argument
parser.add_argument('n_layers',type=int)           # positional argument
parser.add_argument('heads',type=int)           # positional argument
parser.add_argument('droupout',type=float)           # positional argument
parser.add_argument('batch_size',type=int) #batch size
parser.add_argument('data_set',type=str) #path to dataset file with features
parser.add_argument('classification_file',type=str) # file containing classifications for BGCs
parser.add_argument('token_path',type=str) #path to directory with tokens
parser.add_argument('--seed',type=int,default=0) #random seed
parser.add_argument('--model_output',type=str,default="classification") #output name for classifier
parser.add_argument('--epochs',type=int,default=50) #output name for classifier
parser.add_argument('--use_pos_weights',type=int,default=1) #if 0, set all pos_weights to 1
parser.add_argument('--write_metrics',type=int,default=1) #if 0, do not write metrics after training
parser.add_argument('--train_fraction',type=float,default=0.9) #fraction to use for training, remaining will be val
args = parser.parse_args()
torch.manual_seed(args.seed)
unknown_threshold = args.unknown_threshold
max_bgc_length = args.max_bgc_length

training_data_file = open(args.data_set)
classification_file = open(args.classification_file)
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
        if len(classification_types) == 0:
            for i in range(1,len(split_line)):
                classification_types.append(i-1)    
                classification_counts[i-1] = 0
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
for line in training_data_file:
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


pos_weights = []
for c in classification_counts:
    if args.use_pos_weights == 0:
        pos_weights.append(1)
        continue
    if classification_counts[c] != 0:
        pos_weights.append((total_count - classification_counts[c])/classification_counts[c])
    else:
        pos_weights.append((total_count - classification_counts[c])/1)
print(pos_weights)
pos_weights = torch.as_tensor(pos_weights, dtype=torch.float).to(device)


y_vals = []
for name in bgc_names: 
    y_vals.append(bgc_classifications[name])

classification_dataset = BGC_MLM_tools.BGCClassificationDatasets(
   bgc_tokens, token_list, y_vals, seq_len=max_bgc_length)

train_set, val_set = torch.utils.data.random_split(classification_dataset, [math.floor(args.train_fraction*len(bgc_names)), len(bgc_names) - math.floor(args.train_fraction*len(bgc_names))])

if device == "cpu":
    unmasked_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    unmasked_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")

#using pretrained model
classifier_model = BGC_MLM_tools.BGCMultiLabelClassifier(mlm.bgc_mlm,args.d_model,len(y_vals[0]),freeze=False)
bert_trainer = BGC_MLM_tools.BGCMultiLabelTrainier(classifier_model, unmasked_loader, val_set, pos_weights, device=device)   
epochs = args.epochs
classifier_model.to(device)
#print("Checking device for model parameters:")
#for name, param in classifier_model.named_parameters():
#    print(f"Parameter '{name}': Device = {param.device}")

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

if args.write_metrics == 0:
    torch.save(classifier_model.state_dict(), args.model_name + "_" +args.model_output)
    exit()

#check metrics
val_predictions = bert_trainer.predict(val_set,batch_size=args.batch_size)
train_predictions = bert_trainer.predict(train_set,batch_size=args.batch_size)
val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, pin_memory=False)
test_loader = DataLoader(
        train_set, batch_size=1, shuffle=False, pin_memory=False)
print(val_predictions.shape)
for i in range(0, len(classification_types)):
    print(classification_types[i])
    
    #train
    data_iter = enumerate(test_loader)
    true_y = []
    for j, data in data_iter:
        #print(data["classification_label"].shape)
        true_y.append(int(data["classification_label"].detach().cpu().numpy()[0,i]))
    if sum(true_y) < 2:
        print("not enough of class")
        continue
    
    predicted_y = train_predictions.detach().cpu().numpy()[:,i]
    predicted_y_class = [x >0.5 for x in predicted_y]
    accuracy = accuracy_score(true_y, predicted_y_class)
    balanced_accuracy = balanced_accuracy_score(true_y, predicted_y_class)
    precision =precision_score(true_y, predicted_y_class)
    recall = recall_score(true_y, predicted_y_class)
    roc_auc = roc_auc_score(true_y, predicted_y)
    print("train accuracy: " +str(accuracy))
    print("train balanced accuracy: " +str(balanced_accuracy))
    print("train precision: " +str(precision))
    print("train recall: " +str(recall))
    print("train ROC AUC: " +str(roc_auc))
    
    #val
    data_iter = enumerate(val_loader)
    true_y = []
    for j, data in data_iter:
        #print(data["classification_label"].shape)
        true_y.append(int(data["classification_label"].detach().cpu().numpy()[0,i]))
    
    
    
    if sum(true_y) < 2:
        print("not enough of class")
        continue
    predicted_y = val_predictions.detach().cpu().numpy()[:,i]
    predicted_y_class = [x >0.5 for x in predicted_y]
    #print(predicted_y)
    #print(predicted_y_class)
    #predicte_y_classification 
    #print(predicted_y.shape)
    accuracy = accuracy_score(true_y, predicted_y_class)
    balanced_accuracy = balanced_accuracy_score(true_y, predicted_y_class)
    precision =precision_score(true_y, predicted_y_class)
    recall = recall_score(true_y, predicted_y_class)
    roc_auc = roc_auc_score(true_y, predicted_y)
    print("val accuracy: " +str(accuracy))
    print("val balanced accuracy: " +str(balanced_accuracy))
    print("val precision: " +str(precision))
    print("val recall: " +str(recall))
    print("val ROC AUC: " +str(roc_auc))

torch.save(classifier_model.state_dict(), args.model_name + "_" +args.model_output)
