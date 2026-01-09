# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 18:20:22 2025
details on fp types:
https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
https://zoehlerbz.medium.com/representation-of-molecular-fingerprints-with-python-and-rdkit-for-ai-models-8b146bcf3230

@author: Allison Walker
"""

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
import argparse
import os

def write_fp(fp,bgc_id,outfile):
    outfile.write(bgc_id)
    for val in fp:
        outfile.write("," + str(val))
    outfile.write("\n")
    
parser = argparse.ArgumentParser()            
parser.add_argument('dataset_file')
args = parser.parse_args()


fp_sizes = [2048,4096,8192]
fp_gens = {}
outfiles = {}
outfiles_single_fp = {}
for s in fp_sizes:
    fp_gens[s] = {}
    fp_gens[s]["mfpgen"] = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=s)
    fp_gens[s]["rdkgen"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=s)
    fp_gens[s]["apgen"] = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=s)
    fp_gens[s]["ttgen"] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=s)
    fp_gens[s]["fmgen"] = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=s,
                  atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    outfiles[s] = {}
    outfiles_single_fp[s] = {}
    outfiles[s]["mf"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_mf_" + str(s) + ".csv","w")
    outfiles[s]["rdk"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_rdk_" + str(s) + ".csv","w")
    outfiles[s]["ap"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_ap_" + str(s) + ".csv","w")
    outfiles[s]["tt"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_tt_" + str(s) + ".csv","w")
    outfiles[s]["fm"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_fm_" + str(s) + ".csv","w")
    outfiles_single_fp[s]["mf"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_mf_" + str(s) + "single.csv","w")
    outfiles_single_fp[s]["rdk"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_rdk_" + str(s) + "single.csv","w")
    outfiles_single_fp[s]["ap"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_ap_" + str(s) + "single.csv","w")
    outfiles_single_fp[s]["tt"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_tt_" + str(s) + "single.csv","w")
    outfiles_single_fp[s]["fm"] = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_fm_" + str(s) + "single.csv","w")
maccs_outfile = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_maccs.csv","w")
maccs_single_outfile = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_maccs_single.csv","w")
if not os.path.exists(args.dataset_file):
    print("Input file does not exist")
    exit()

for line in open(args.dataset_file):
    split_line = line.split(",")
    if len(split_line) ==1:
        continue
    bgc_id = split_line[0]
    first_smiles = split_line[1]
    mol = Chem.MolFromSmiles(first_smiles)
    fingerprints = {}
    for s in fp_sizes:
        fingerprints[s] = {}
        fingerprints[s]["mfp"] = fp_gens[s]["mfpgen"].GetFingerprint(mol)
        fingerprints[s]["rdk"] = fp_gens[s]["rdkgen"].GetFingerprint(mol)
        fingerprints[s]["ap"] = fp_gens[s]["apgen"].GetFingerprint(mol)
        fingerprints[s]["tt"] = fp_gens[s]["ttgen"].GetFingerprint(mol)
        fingerprints[s]["fm"] = fp_gens[s]["fmgen"].GetFingerprint(mol)
        write_fp(fp_gens[s]["mfpgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["mf"])
        write_fp(fp_gens[s]["rdkgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["rdk"])
        write_fp(fp_gens[s]["apgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["ap"])
        write_fp(fp_gens[s]["ttgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["tt"])
        write_fp(fp_gens[s]["fmgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["fm"])
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    write_fp(MACCSkeys.GenMACCSKeys(mol),bgc_id,maccs_single_outfile)
    for i in range(2,len(split_line)):
        smiles = split_line[i]
        mol = Chem.MolFromSmiles(smiles)
        for s in fp_sizes:
            fingerprints[s]["mfp"] = fingerprints[s]["mfp"] | fp_gens[s]["mfpgen"].GetFingerprint(mol)
            fingerprints[s]["rdk"] = fingerprints[s]["rdk"] | fp_gens[s]["rdkgen"].GetFingerprint(mol)
            fingerprints[s]["ap"] = fingerprints[s]["ap"] | fp_gens[s]["apgen"].GetFingerprint(mol)
            fingerprints[s]["tt"] = fingerprints[s]["tt"] | fp_gens[s]["ttgen"].GetFingerprint(mol)
            fingerprints[s]["fm"] = fingerprints[s]["fm"] | fp_gens[s]["fmgen"].GetFingerprint(mol)
            write_fp(fp_gens[s]["mfpgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["mf"])
            write_fp(fp_gens[s]["rdkgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["rdk"])
            write_fp(fp_gens[s]["apgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["ap"])
            write_fp(fp_gens[s]["ttgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["tt"])
            write_fp(fp_gens[s]["fmgen"].GetFingerprint(mol),bgc_id,outfiles_single_fp[s]["fm"])
        maccs_fp =  maccs_fp | MACCSkeys.GenMACCSKeys(mol)
        write_fp(MACCSkeys.GenMACCSKeys(mol),bgc_id,maccs_single_outfile)
    for s in fp_sizes:
        write_fp(fingerprints[s]["mfp"],bgc_id,outfiles[s]["mf"])
        write_fp(fingerprints[s]["rdk"],bgc_id,outfiles[s]["rdk"])
        write_fp(fingerprints[s]["ap"],bgc_id,outfiles[s]["ap"])
        write_fp(fingerprints[s]["tt"],bgc_id,outfiles[s]["tt"])
        write_fp(fingerprints[s]["fm"],bgc_id,outfiles[s]["fm"])
    write_fp(maccs_fp,bgc_id,maccs_outfile)

for s in fp_sizes:
    outfiles[s]["mf"].close()
    outfiles[s]["rdk"].close()
    outfiles[s]["ap"].close()
    outfiles[s]["fm"].close()
maccs_outfile.close()