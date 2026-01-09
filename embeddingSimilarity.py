# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 15:19:56 2025

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
from sklearn.metrics import classification_report
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys

parser = argparse.ArgumentParser()            
parser.add_argument('model_name')           # positional argument
parser.add_argument('model_param_file')           # positional argument
parser.add_argument('infile',type=str) #list of BGC ids for which to calculate similarity
parser.add_argument('token_path',type=str) #path to directory with tokens from genome
parser.add_argument('outfile',type=str) #outfile prefix for similarity calculaiton
parser.add_argument('--seed',type=int,default=0) #random seed
parser.add_argument('--fp_size',type=int,default=8192) #fraction to use for training, remaining will be val
args = parser.parse_args()
torch.manual_seed(args.seed)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token_list = []
token_list_file = open("token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt")
for line in token_list_file:
    token_list.append(line.replace("\n",""))
    
bgc_tokens = []
bgc_names = []

i= 0
for line in open(args.infile):
    filename = line.replace("\n","")
    
    bgc_name = filename
    if not os.path.exists(args.token_path + "/" + filename):
        continue
    infile = open(args.token_path + "/" + filename)
    bgc_token = ["CLS"]
    useBGC = True
    for line in infile:
        split_line = line.split(",")
    if len(split_line) + 2 > max_bgc_length:
        print("WARNING: BGC " + bgc_name + " longer than length limit, " + str(len(split_line) +2) +" , only using middle of BGC")
        split_line = split_line[int(len(split_line)/2)-49:int(len(split_line)/2)+49]
        print(len(split_line))
    for t in split_line:
        if t in token_list:
            bgc_token.append(t)
        else:
            bgc_token.append("UNK")
        if len(bgc_token) + 1 == max_bgc_length:
            break
    bgc_token.append("SEP")
    if len(bgc_token) < max_bgc_length:
        j = len(bgc_token)
        while j < max_bgc_length:
           bgc_token.append("PAD")
           j += 1
    bgc_tokens.append(bgc_token)
    bgc_names.append(bgc_name)
    i += 1

fps = []
y = []
fp_size = 8192
for i in range(0, fp_size):
    y.append(0)
for name in bgc_names: 
    fps.append(y)
    
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
target_embedding_model = BGC_MLM_tools.BGCMetricLearning(mlm.bgc_mlm,d_model,args.fp_size)
target_embedding_model.load_state_dict(torch.load(args.model_name,weights_only=True, map_location=torch.device(device)))
target_embedding_model.to(device)
dataset = BGC_MLM_tools.BGCClassificationDatasets(
   bgc_tokens, token_list, fps, seq_len=max_bgc_length)

if device == "cpu":
    unmasked_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    unmasked_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")

bert_trainer = BGC_MLM_tools.BGCMetricTrainer(target_embedding_model, unmasked_loader, dataset, None, device=device)   
embedding = bert_trainer.predict(dataset,batch_size=batch_size)

model_name = args.model_name[args.model_name.rfind("/")+1:len(args.model_name)]
outfile = open(args.outfile + model_name +  "_embed_sim.csv",'w')
i = 0

for bgc1 in bgc_names:
    j = 0
    prediction1 = embedding[i,:]
    for bgc2 in bgc_names:
        if bgc1 == bgc2:
            j += 1
            continue
        prediction2 = embedding[j,:]
        cos_sim = np.dot(prediction1.cpu(), prediction2.cpu()) / (np.linalg.norm(prediction1.cpu()) * np.linalg.norm(prediction2.cpu()))
        outfile.write(bgc1 +"," + bgc2 + "," + str(cos_sim) +"\n")
        j += 1
    i += 1
outfile.close()