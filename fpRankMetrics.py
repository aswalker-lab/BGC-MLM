# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 09:57:28 2025

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
from sklearn.metrics import classification_report
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

def list_to_bitvect(bits):
    bv = ExplicitBitVect(len(bits))
    for i, bit in enumerate(bits):
        if bit == 1:
            bv.SetBit(i)
    return bv

def find_rank(bgc_name,sorted_dic,fps):
    bgcs_ranked_above = []
    bgc1_fps = []
    correct_score = 0
    for fp in fps[bgc_name]:
        bgc1_fps.append(list_to_bitvect(fp))
    for i in range(0,len(sorted_dic)):
        bgc2 = sorted_dic[i][0].split("_")[0]
        if bgc2 not in bgcs_ranked_above:
            bgcs_ranked_above.append(bgc2)
        if bgc_name == bgc2:
            correct_score = sorted_dic[i][1]
            break
        found_match = False
        for fp1 in bgc1_fps:
            for fp2 in fps[bgc2]:
                if DataStructs.TanimotoSimilarity(fp1, list_to_bitvect(fp2)) == 1:
                    found_match = True
                    correct_score = sorted_dic[i][1]
                    break
            if found_match:
                break
        if found_match:
            break
    return len(bgcs_ranked_above), correct_score



torch.manual_seed(args.seed)
unknown_threshold = args.unknown_threshold
max_bgc_length = args.max_bgc_length

test_data_file = open(args.data_set)
classification_file = open(args.comparison_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token_list = []
token_list_file = open("token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt")
for line in token_list_file:
    token_list.append(line.replace("\n",""))

test_bgc_tokens = []
test_bgc_names = []

comparison_bgc_tokens = []
comparison_bgc_names = []

classification_types = []
classification_counts = {}
bgc_fps = {}
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
        if len(classification_types) == 0:
            #if no header then number classifications
            for i in range(1,len(split_line)):
                classification_types.append(i-1)    
                classification_counts[i-1] = 0
        if bgc_name not in bgc_fps:
            bgc_fps[bgc_name] = []
        
        i = 0
        fp = []
        for val in split_line[1:len(split_line)]:
        #for val in split_line[1:4]:
            fp.append(int(val))
            if int(val) == 1:
                classification_counts[classification_types[i]] += 1
                total_count += 1
            i += 1
        bgc_fps[bgc_name].append(fp)
        if bgc_name in comparison_bgc_tokens:
            continue
        infile = open(args.token_path + "/" + bgc_name)
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
        comparison_bgc_names.append(bgc_name)
        comparison_bgc_tokens.append(bgc_token)
i= 0
for line in test_data_file:
    split_line = line.split(",")
    filename = split_line[0]
    
    if filename not in bgc_fps:
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
    test_bgc_names.append(bgc_name)           
    test_bgc_tokens.append(bgc_token)
    
    i += 1

test_y_vals = []
comparison_y_vals = []
for name in test_bgc_names: 
    y = []
    for i in range(0,len(bgc_fps[name][0])):
        y.append(0)
    test_y_vals.append(y)

for name in comparison_bgc_names:
    y = []
    for i in range(0,len(bgc_fps[name][0])):
        y.append(0)
    comparison_y_vals.append(y)

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

classification_model = BGC_MLM_tools.BGCMultiLabelClassifier(mlm.bgc_mlm,args.d_model,len(test_y_vals[0]),freeze=True)
classification_model.load_state_dict(torch.load(args.model_name,weights_only=True, map_location=torch.device(device)))
classification_model.to(device)

test_dataset = BGC_MLM_tools.BGCClassificationDatasets(
   test_bgc_tokens, token_list, test_y_vals, seq_len=max_bgc_length)

comparison_dataset = BGC_MLM_tools.BGCClassificationDatasets(
   comparison_bgc_tokens, token_list, comparison_y_vals, seq_len=max_bgc_length)

if device == "cpu":
    test_unmasked_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
    comparison_unmasked_loader = DataLoader(
        comparison_dataset, batch_size=1, shuffle=False, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    test_unmasked_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")
    comparison_unmasked_loader = DataLoader(
        comparison_dataset, batch_size=1, shuffle=False, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")

print("predictions 1")
test_bert_trainer = BGC_MLM_tools.BGCMultiLabelTrainier(classification_model, test_unmasked_loader, test_dataset, None, device=device)   
test_predictions = test_bert_trainer.predict(test_dataset,batch_size=args.batch_size)

print("predictions 2")
comparison_bert_trainer = BGC_MLM_tools.BGCMultiLabelTrainier(classification_model, comparison_unmasked_loader, comparison_dataset, None, device=device)   
comparison_predictions = comparison_bert_trainer.predict(comparison_dataset,batch_size=args.batch_size)

if "/" not in args.comparison_set:
    outprefix = args.data_set[0:args.data_set.find(".")] + "_" + args.comparison_set[0:args.comparison_set.find(".")]
else:
    outprefix = args.data_set[0:args.data_set.find(".")] + "_" + args.comparison_set[args.comparison_set.rfind("/")+1:args.comparison_set.find(".")]
if not os.path.isfile(outprefix + "_rankings_bgc_to_mol.txt"):
    outfile_bgc_to_mol = open(outprefix +  "_rankings_bgc_to_mol.txt",'w')
    outfile_bgc_to_mol.write("model_name,avg_cos_rank,top_5_cos,top_10_cos,avg_cos_correct,avg_tani_rank,top_5_tani,top_10_tani,avg_tani_correct\n")
else:
    outfile_bgc_to_mol = open(outprefix + "_rankings_bgc_to_mol.txt",'a')

if not os.path.isfile(outprefix + "_rankings_mol_to_bgc.txt"):
    outfile_mol_to_bgc = open(outprefix +  "_rankings_mol_to_bgc.txt",'w')
    outfile_mol_to_bgc.write("model_name,avg_cos_rank,top_5_cos,top_10_cos,avg_cos_correct,avg_tani_rank,avg_tani_correct\n")
else:
    outfile_mol_to_bgc = open(outprefix + "_rankings_mol_to_bgc.txt",'a')

#Ranking of correct moleucules (BGC to molecule)
avg_top_rank_cos = 0
avg_top_rank_tanimoto = 0
avg_cos_correct = 0
avg_tani_correct = 0
top_5_correct_cos = 0
top_10_correct_cos = 0
top_5_correct_tani = 0
top_10_correct_tani = 0
print("BGC to mols")
for i in range(0,len(test_bgc_names)):
    bgc_name = test_bgc_names[i]
    prediction = test_predictions[i,:]
    prediction_tresholded = prediction > 0.5
    tani_scores = {}
    cos_scores = {}
    for bgc2 in bgc_fps:
        j = 0
        for fp in bgc_fps[bgc2]:
            cos_sim = np.dot(prediction.cpu(), fp) / (np.linalg.norm(prediction.cpu()) * np.linalg.norm(fp))
            tanimoto_similarity = DataStructs.TanimotoSimilarity(list_to_bitvect(prediction_tresholded.tolist()), list_to_bitvect(fp))
            tani_scores[bgc2+"_"+str(j)] = tanimoto_similarity
            cos_scores[bgc2+"_"+str(j)] = cos_sim
            j += 1
    sorted_by_tanimoto = sorted(tani_scores.items(), key=lambda item: item[1],reverse=True)
    sorted_by_cos = sorted(cos_scores.items(), key=lambda item: item[1],reverse=True)
    tani_rank, tani_correct_score = find_rank(bgc_name,sorted_by_tanimoto,bgc_fps)
    cos_rank, cos_correct_score = find_rank(bgc_name,sorted_by_cos,bgc_fps)
    avg_top_rank_cos += cos_rank
    avg_top_rank_tanimoto += tani_rank
    avg_cos_correct += cos_correct_score
    avg_tani_correct += tani_correct_score
    if cos_rank <= 5:
        top_5_correct_cos += 1
    if cos_rank <= 10:
        top_10_correct_cos += 1
    if tani_rank <= 5:
        top_5_correct_tani += 1
    if cos_rank <=10:
        top_10_correct_tani += 1
avg_top_rank_cos /= len(test_bgc_names)
avg_top_rank_tanimoto /= len(test_bgc_names)
avg_cos_correct /= len(test_bgc_names)
avg_tani_correct /= len(test_bgc_names)
top_5_correct_cos /= len(test_bgc_names)
top_10_correct_cos /= len(test_bgc_names)
top_5_correct_tani /= len(test_bgc_names)
top_10_correct_tani /= len(test_bgc_names)
outfile_bgc_to_mol.write(args.model_name)
outfile_bgc_to_mol.write("," + str(avg_top_rank_cos))
outfile_bgc_to_mol.write("," + str(top_5_correct_cos))
outfile_bgc_to_mol.write("," + str(top_10_correct_cos))
outfile_bgc_to_mol.write("," + str(avg_cos_correct))
outfile_bgc_to_mol.write("," + str(avg_top_rank_tanimoto))
outfile_bgc_to_mol.write("," + str(top_5_correct_tani))
outfile_bgc_to_mol.write("," + str(top_10_correct_tani))
outfile_bgc_to_mol.write("," + str(avg_tani_correct)+"\n")
outfile_bgc_to_mol.close()

    
    
#Ranking of correct BGCs (molecule to BGC)
total_molecules = 0
avg_top_rank_cos = 0
avg_top_rank_tanimoto = 0
avg_cos_correct = 0
avg_tani_correct = 0
top_5_correct_cos = 0
top_10_correct_cos = 0
top_5_correct_tani = 0
top_10_correct_tani = 0

print("mols to BGC")
for i in range(0,len(test_bgc_names)):
    bgc_name = test_bgc_names[i]
    fps = bgc_fps[bgc_name]
    tani_scores = {}
    cos_scores = {}
    for fp in fps:
        total_molecules += 1
        fp_bit = list_to_bitvect(fp)
        for j in range(0, len(comparison_bgc_names)):
            bgc2 = comparison_bgc_names[j]
            prediction = comparison_predictions[j,:]
            prediction_tresholded = prediction > 0.5
            cos_sim = np.dot(prediction.cpu(), fp) / (np.linalg.norm(prediction.cpu()) * np.linalg.norm(fp))
            tanimoto_similarity = DataStructs.TanimotoSimilarity(list_to_bitvect(prediction_tresholded.tolist()), fp_bit)
            tani_scores[bgc2] = tanimoto_similarity
            cos_scores[bgc2] = cos_sim
        sorted_by_tanimoto = sorted(tani_scores.items(), key=lambda item: item[1],reverse=True)
        sorted_by_cos = sorted(cos_scores.items(), key=lambda item: item[1],reverse=True)
        tani_rank, tani_correct_score = find_rank(bgc_name,sorted_by_tanimoto,bgc_fps)
        cos_rank, cos_correct_score = find_rank(bgc_name,sorted_by_cos,bgc_fps)
        avg_top_rank_cos += cos_rank
        avg_top_rank_tanimoto += tani_rank
        avg_cos_correct += cos_correct_score
        avg_tani_correct += tani_correct_score
        if cos_rank <= 5:
            top_5_correct_cos += 1
        if cos_rank <= 10:
            top_10_correct_cos += 1
        if tani_rank <= 5:
            top_5_correct_tani += 1
        if cos_rank <=10:
            top_10_correct_tani += 1
avg_top_rank_cos /= total_molecules
top_5_correct_cos /= total_molecules
top_10_correct_cos /= total_molecules
top_5_correct_tani /= total_molecules
top_10_correct_tani /= total_molecules
avg_top_rank_tanimoto /= total_molecules
avg_cos_correct /= total_molecules
avg_tani_correct /= total_molecules

outfile_mol_to_bgc.write(args.model_name)
outfile_mol_to_bgc.write("," + str(avg_top_rank_cos))
outfile_mol_to_bgc.write("," + str(top_5_correct_cos))
outfile_mol_to_bgc.write("," + str(top_10_correct_cos))
outfile_mol_to_bgc.write("," + str(avg_cos_correct))
outfile_mol_to_bgc.write("," + str(avg_top_rank_tanimoto))
outfile_mol_to_bgc.write("," + str(top_5_correct_tani))
outfile_mol_to_bgc.write("," + str(top_10_correct_tani))
outfile_mol_to_bgc.write("," + str(avg_tani_correct)+"\n")
outfile_mol_to_bgc.close()
