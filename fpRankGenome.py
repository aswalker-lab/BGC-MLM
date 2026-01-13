# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:59:22 2025

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
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
from src.utils import arg_parse
from sys import argv

args = arg_parse.parse_args("fp_rank_genome", argv[1:])
def list_to_bitvect(bits):
    bv = ExplicitBitVect(len(bits))
    for i, bit in enumerate(bits):
        if bit == 1:
            bv.SetBit(i)
    return bv


torch.manual_seed(args.seed)


# read model parameters
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

mol = Chem.MolFromSmiles(args.smiles)
if args.fp_type == "morgan":
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=args.fp_size)
elif args.fp_type == "rdkit":
    fp_generator =rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=args.fp_size)
elif args.fp_type == "ap":
    fp_generator =rdFingerprintGenerator.GetAtomPairGenerator(fpSize=args.fp_size)
elif args.fp_type == "tt":
    rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=args.fp_size)
elif args.fp_type == "functional_morgan":
    fp_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(radius=2,fpSize=args.fp_size,
                  atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
elif args.fp_type == "maccs":
    fp = MACCSkeys.GenMACCSKeys(mol)
else:
    print("Fingerprint type not recognized")
    exit()

if args.fp_type != "maccs":
    fp = fp_generator.GetFingerprint(mol)


genome_bgc_tokens = []
genome_bgc_names = []

i= 0
for filename in os.listdir(args.token_path):
    bgc_name = filename[filename.find("/")+1:len(filename)]
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
    genome_bgc_tokens.append(bgc_token)
    genome_bgc_names.append(bgc_name)
    i += 1

y_vals = []
y = []
for i in range(0, len(fp)):
    y.append(0)
for name in genome_bgc_names: 
    y_vals.append(y)

# load model from file
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
print(len(fp))
print(len(y_vals[0]))
classification_model = BGC_MLM_tools.BGCMultiLabelClassifier(mlm.bgc_mlm,d_model,len(y_vals[0]),freeze=True)
classification_model.load_state_dict(torch.load(args.model_name,weights_only=True, map_location=torch.device(device)))
classification_model.to(device)

genome_dataset = BGC_MLM_tools.BGCClassificationDatasets(
   genome_bgc_tokens, token_list, y_vals, seq_len=max_bgc_length)

if device == "cpu":
    genome_unmasked_loader = DataLoader(
        genome_dataset, batch_size=1, shuffle=False, pin_memory=True)#pin_memory=False, pin_memory_device="cuda")
else:
    genome_unmasked_loader = DataLoader(
        genome_dataset, batch_size=1, shuffle=False, pin_memory=False)#pin_memory=False, pin_memory_device="cuda")

genome_bert_trainer = BGC_MLM_tools.BGCMultiLabelTrainier(classification_model, genome_unmasked_loader, genome_dataset, None, device=device)   
genome_predictions = genome_bert_trainer.predict(genome_dataset,batch_size=batch_size)

tani_scores = {}
cos_scores = {}
for i in range(0,len(genome_bgc_names)):
    bgc_name = genome_bgc_names[i]
    prediction = genome_predictions[i,:]
    prediction_tresholded = prediction > 0.5
    cos_sim = np.dot(prediction.cpu(), fp) / (np.linalg.norm(prediction.cpu()) * np.linalg.norm(fp))
    tanimoto_similarity = DataStructs.TanimotoSimilarity(list_to_bitvect(prediction_tresholded.tolist()), list_to_bitvect(fp))
    tani_scores[bgc_name] = tanimoto_similarity
    cos_scores[bgc_name] = cos_sim
sorted_by_tanimoto = sorted(tani_scores.items(), key=lambda item: item[1],reverse=True)
sorted_by_cos = sorted(cos_scores.items(), key=lambda item: item[1],reverse=True)
outfile = open(args.outfile,'w')
outfile.write("Cosine ranking\n")
for item in sorted_by_cos:
    outfile.write(item[0] + "," + str(item[1]) + "\n")
outfile.write("Tanimoto ranking\n")
for item in sorted_by_tanimoto:
    outfile.write(item[0] + "," + str(item[1]) + "\n")
outfile.close()
