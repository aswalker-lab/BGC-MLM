# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 09:14:00 2025

@author: Allison Walker
This script calculates min and max molecular descriptors from SMIILES for
downstream descriptor prediction tasks
"""

import argparse
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

def newMinValue(curr_val, curr_min):
    if curr_val < curr_min:
        return curr_val
    else:
        return curr_min

def newMaxValue(curr_val, curr_max):
    if curr_val > curr_max:
        return curr_val
    else:
        return curr_max


parser = argparse.ArgumentParser()            
parser.add_argument('dataset_file')
args = parser.parse_args()

if not os.path.exists(args.dataset_file):
    print("Input file does not exist")
    exit()

exact_mass_min_dic = {}
exact_mass_max_dic = {}
tpsa_min_dic = {}
tpsa_max_dic = {}
fsp3c_min_dic = {}
fsp3c_max_dic = {}
aliphatic_rings_min_dic = {}
aliphatic_rings_max_dic = {}
aromatic_rings_min_dic = {}
aromatic_rings_max_dic = {}
h_acceptors_min_dic = {}
h_acceptors_max_dic = {}
h_donors_min_dic = {}
h_donors_max_dic = {}
nh2_min_dic = {}
nh2_max_dic = {}
coo_min_dic = {}
coo_max_dic = {}
amide_min_dic = {}
amide_max_dic = {}
halogen_min_dic = {}
halogen_max_dic = {}
lactam_min_dic = {}
lactam_max_dic = {}
lactone_min_dic = {}
lactone_max_dic = {}
nitro_min_dic = {}
nitro_max_dic = {}
oxazole_min_dic = {}
oxazole_max_dic = {}
oxime_min_dic = {}
oxime_max_dic = {}
pyridine_min_dic = {}
pyridine_max_dic ={}
thiazole_min_dic = {}
thiazole_max_dic = {}
urea_min_dic = {}
urea_max_dic = {}
clogp_min_dic = {}
clogp_max_dic = {}
for line in open(args.dataset_file):
    split_line = line.split(",")
    if len(split_line) ==1:
        continue
    bgc_id = split_line[0]
    tpsa_max = 0
    exact_mass_max = 0
    fsp3c_max = 0
    aliphatic_rings_max = 0
    aromatic_rings_max = 0
    h_acceptors_max = 0
    h_donors_max = 0
    nh2_max = 0
    coo_max = 0
    amide_max = 0
    halogen_max = 0
    lactam_max = 0
    lactone_max = 0
    nitro_max = 0
    oxazole_max = 0
    oxime_max = 0
    pyridine_max = 0
    thiazole_max = 0
    urea_max = 0
    clogp_max = 0
    for i in range(1,len(split_line)):
        smiles = split_line[i]
        mol = Chem.MolFromSmiles(smiles)
        exact_mass = Descriptors.ExactMolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        fsp3c = Descriptors.FractionCSP3(mol)
        aliphatic_rings = Descriptors.NumAliphaticRings(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        h_donors = Descriptors.NumHDonors(mol)
        #other fragments that can be calced: https://rdkit.org/docs/source/rdkit.Chem.Fragments.html
        nh2 = Descriptors.fr_NH2(mol)
        coo = Descriptors.fr_COO(mol)
        amide =Descriptors.fr_amide(mol)
        halogen = Descriptors.fr_halogen(mol)
        lactam = Descriptors.fr_lactam(mol)
        lactone = Descriptors.fr_lactone(mol)
        nitro = Descriptors.fr_nitro(mol)
        oxazole = Descriptors.fr_oxazole(mol)
        oxime = Descriptors.fr_oxime(mol)
        pyridine = Descriptors.fr_pyridine(mol)
        thiazole = Descriptors.fr_thiazole(mol)
        urea = Descriptors.fr_urea(mol)
        clogp = Chem.Crippen.MolLogP(mol)
        if i == 1:
            exact_mass_min = exact_mass
            tpsa_min = tpsa
            fsp3c_min = fsp3c
            aliphatic_rings_min = aliphatic_rings
            aromatic_rings_min = aromatic_rings
            h_acceptors_min = h_acceptors
            h_donors_min = h_donors
            nh2_min = nh2
            coo_min = coo
            amide_min = amide
            halogen_min = halogen
            lactam_min = lactam
            lactone_min = lactone
            nitro_min = nitro
            oxazole_min = oxazole
            oxime_min = oxime
            pyridine_min = pyridine
            thiazole_min = thiazole
            urea_min = urea
            clogp_min = clogp
        #min values
        exact_mass_min = newMinValue(exact_mass, exact_mass_min)
        tpsa_min = newMinValue(tpsa,tpsa_min)
        fsp3c_min = newMinValue(fsp3c,fsp3c_min)
        aliphatic_rings_min = newMinValue(aliphatic_rings, aliphatic_rings_min)
        aromatic_rings_min = newMinValue(aromatic_rings, aromatic_rings_min)
        h_acceptors_min = newMinValue(h_acceptors, h_acceptors_min)
        h_donors_min = newMinValue(h_donors,h_donors_min)
        nh2_min = newMinValue(nh2, nh2_min)
        coo_min = newMinValue(coo,coo_min)
        amide_min = newMinValue(amide,amide_min)
        halogen_min = newMinValue(halogen, halogen_min)
        lactam_min = newMinValue(lactam,lactam_min)
        lactone_min = newMinValue(lactone,lactone_min)
        nitro_min = newMinValue(nitro,nitro_min)
        oxazole_min = newMinValue(oxazole,oxazole_min)
        oxime_min = newMinValue(oxime,oxime_min)
        pyridine_min = newMinValue(pyridine,pyridine_min)
        thiazole_min = newMinValue(thiazole,thiazole_min)
        urea_min = newMinValue(urea,urea_min)
        clogp_min = newMinValue(clogp,clogp_min)            
        #max values
        exact_mass_max = newMaxValue(exact_mass,exact_mass_max)
        tpsa_max = newMaxValue(tpsa,tpsa_max)
        fsp3c_max = newMaxValue(fsp3c,fsp3c_max)
        aliphatic_rings_max = newMaxValue(aliphatic_rings,aliphatic_rings_max)
        aromatic_rings_max = newMaxValue(aromatic_rings,aromatic_rings_max)
        h_acceptors_max = newMaxValue(h_acceptors,h_acceptors_max)
        h_donors_max = newMaxValue(h_donors,h_donors_max)
        nh2_max = newMaxValue(nh2,nh2_max)
        coo_max = newMaxValue(coo,coo_max)
        amide_max = newMaxValue(amide,amide_max)
        halogen_max = newMaxValue(halogen,halogen_max)
        lactam_max = newMaxValue(lactam,lactam_max)
        lactone_max = newMaxValue(lactone,lactone_max)
        nitro_max = newMaxValue(nitro,nitro_max)
        oxazole_max = newMaxValue(oxazole,oxazole_max)
        oxime_max = newMaxValue(oxime,oxime_max)
        pyridine_max = newMaxValue(pyridine,pyridine_max)
        thiazole_max = newMaxValue(thiazole,thiazole_max)
        urea_max = newMaxValue(urea,urea_max)
        clogp_max = newMaxValue(clogp,clogp_max)
        
    exact_mass_min_dic[bgc_id] = exact_mass_min
    exact_mass_max_dic[bgc_id] = exact_mass_max
    tpsa_min_dic[bgc_id] = tpsa_min
    tpsa_max_dic[bgc_id] = tpsa_max
    fsp3c_min_dic[bgc_id] = fsp3c_min
    fsp3c_max_dic[bgc_id] = fsp3c_max
    aliphatic_rings_min_dic[bgc_id] = aliphatic_rings_min
    aliphatic_rings_max_dic[bgc_id] = aliphatic_rings_max
    aromatic_rings_min_dic[bgc_id] = aromatic_rings_min
    aromatic_rings_max_dic[bgc_id] = aromatic_rings_max
    h_acceptors_min_dic[bgc_id] = h_acceptors_min
    h_acceptors_max_dic[bgc_id] = h_acceptors_max
    h_donors_min_dic[bgc_id] = h_donors_min
    h_donors_max_dic[bgc_id] = h_donors_max
    nh2_min_dic[bgc_id] = nh2_min
    nh2_max_dic[bgc_id] = nh2_max
    coo_min_dic[bgc_id] = coo_min
    coo_max_dic[bgc_id] = coo_max
    amide_min_dic[bgc_id] = amide_min
    amide_max_dic[bgc_id] = amide_max
    halogen_min_dic[bgc_id] = halogen_min
    halogen_max_dic[bgc_id] = halogen_max
    lactam_min_dic[bgc_id] = lactam_min
    lactam_max_dic[bgc_id] = lactam_max
    lactone_min_dic[bgc_id] = lactone_min
    lactone_max_dic[bgc_id] = lactone_max
    nitro_min_dic[bgc_id] = nitro_min
    nitro_max_dic[bgc_id] = nitro_max
    oxazole_min_dic[bgc_id] = oxazole_min
    oxazole_max_dic[bgc_id] = oxazole_max
    oxime_min_dic[bgc_id] = oxime_min    
    oxime_max_dic[bgc_id] = oxime_max
    pyridine_min_dic[bgc_id] = pyridine_min
    pyridine_max_dic[bgc_id] = pyridine_max
    thiazole_min_dic[bgc_id] = thiazole_min
    thiazole_max_dic[bgc_id] = thiazole_max
    urea_min_dic[bgc_id] = urea_min
    urea_max_dic[bgc_id] = urea_max
    clogp_min_dic[bgc_id] = clogp_min
    clogp_max_dic[bgc_id] = clogp_max
    
outfile = open(args.dataset_file[0:args.dataset_file.rfind(".")] + "_descriptors.csv",'w')
outfile.write("bgc_id,min_exact_mass,max_exact_mass,min_tpsa,max_tpsa,")
outfile.write("fsp3c_min,fsp3c_max,")
outfile.write("aliphatic_rings_min,aliphatic_rings_max,")
outfile.write("aromatic_rings_min,aromatic_rings_max,")
outfile.write("h_acceptors_min,h_accemtors_max,")
outfile.write("h_donors_min,h_donors_max,")
outfile.write("nh2_min,nh2_max,")
outfile.write("coo_min,coo_max,")
outfile.write("amide_min,amide_max,")
outfile.write("halogen_min,halogen_max,")
outfile.write("lactam_min,lactam_max,")
outfile.write("lactone_min,lactone_max,")
outfile.write("nitro_min,ntiro_max,")
outfile.write("oxazole_min,oxazole_max,")
outfile.write("oxime_min,oxime_max,")
outfile.write("pyridine_min,pyridine_max,")
outfile.write("thiazole_min,thiazole_max,")
outfile.write("urea_min,urea_max,")
outfile.write("clogp_min,clogp_max\n")
for bgc in exact_mass_min_dic:
    outfile.write(bgc + ",")
    outfile.write(str(exact_mass_min_dic[bgc]) + ",")
    outfile.write(str(exact_mass_max_dic[bgc]) + ",")
    outfile.write(str(tpsa_min_dic[bgc]) + ",")
    outfile.write(str(tpsa_max_dic[bgc]) + ",")
    outfile.write(str(fsp3c_min_dic[bgc]) + ",")
    outfile.write(str(fsp3c_max_dic[bgc]) + ",")
    outfile.write(str(aliphatic_rings_min_dic[bgc]) + ",")
    outfile.write(str(aliphatic_rings_max_dic[bgc]) + ",")
    outfile.write(str(aromatic_rings_min_dic[bgc]) + ",")
    outfile.write(str(aromatic_rings_max_dic[bgc])+",")
    outfile.write(str(h_acceptors_min_dic[bgc]) + ",")
    outfile.write(str(h_acceptors_max_dic[bgc]) + ",")
    outfile.write(str(h_donors_min_dic[bgc]) + ",")
    outfile.write(str(h_donors_max_dic[bgc]) + ",")
    outfile.write(str(nh2_min_dic[bgc]) + ",")
    outfile.write(str(nh2_max_dic[bgc]) + ",")
    outfile.write(str(coo_min_dic[bgc])+",")
    outfile.write(str(coo_max_dic[bgc])+",")
    outfile.write(str(amide_min_dic[bgc])+",")
    outfile.write(str(amide_max_dic[bgc])+",")
    outfile.write(str(halogen_min_dic[bgc])+",")
    outfile.write(str(halogen_max_dic[bgc])+",")
    outfile.write(str(lactam_min_dic[bgc])+",")
    outfile.write(str(lactam_max_dic[bgc])+",")
    outfile.write(str(lactone_min_dic[bgc])+",")
    outfile.write(str(lactone_max_dic[bgc])+",")
    outfile.write(str(nitro_min_dic[bgc])+",")
    outfile.write(str(nitro_max_dic[bgc])+",")
    outfile.write(str(oxazole_min_dic[bgc])+",")
    outfile.write(str(oxazole_max_dic[bgc])+",")
    outfile.write(str(oxime_min_dic[bgc])+",")
    outfile.write(str(oxime_max_dic[bgc])+",")
    outfile.write(str(pyridine_min_dic[bgc])+",")
    outfile.write(str(pyridine_max_dic[bgc])+",")
    outfile.write(str(thiazole_min_dic[bgc])+",")
    outfile.write(str(thiazole_max_dic[bgc])+",")
    outfile.write(str(urea_min_dic[bgc])+",")
    outfile.write(str(urea_max_dic[bgc])+",")
    outfile.write(str(clogp_min_dic[bgc])+",")
    outfile.write(str(clogp_max_dic[bgc])+"\n")
outfile.close()