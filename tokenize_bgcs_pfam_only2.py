# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:29:11 2023

@author: allis
"""
import os
from Bio import SeqIO

in_dir = "./other_refseq/"
outdir = "tokenized_bgcs/pfam/"
outdir2 = "tokenized_bgcs/pfam2/"


def readAntismash5(as_features):
    score_cutoff = 20
    token_list = []
    for feature in as_features:
        if feature.type == "PFAM_domain":
            score = float(feature.qualifiers["score"][0])
            if score <score_cutoff:
                continue
            domain_description = feature.qualifiers["description"][0]
            pfam_id = feature.qualifiers["db_xref"][0]
            pfam_id = pfam_id[pfam_id.find(" ")+1:len(pfam_id)]
            #print(pfam_id[0:pfam_id.find(".")])
            token_list.append(pfam_id[0:pfam_id.find(".")])
    return token_list

def getAllClusterListsInDirectory(path, files):
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            #print join(path, f)
            if ("cluster" in f or "region" in f) and (".gbk" in f or ".gb" in f or ".gbff" in f):
                fullpath = os.path.join(path, f)
                genome_name = fullpath[0:fullpath.rfind("/")]
                genome_name = genome_name[genome_name.rfind("/")+1:len(genome_name)]
                #check if output exists
                if not os.path.exists(outdir + genome_name + "_" +  f[f.rfind("/")+1:f.rfind(".")]): 
                    if not os.path.exists(outdir2 + genome_name + "_" +  f[f.rfind("/")+1:f.rfind(".")]): 
                        files.append((genome_name, os.path.join(path, f)))
            #print files
        else:
            if "genometools" in f or "glimmer" in f or "slurm" in f:
                continue
            files = getAllClusterListsInDirectory(os.path.join(path, f), files)
    return files




files = getAllClusterListsInDirectory(in_dir, [])
for genome_name, f in files:
    record = SeqIO.read(open(f, 'rU'),"genbank")
    #check if cluster is on the contig edge
    on_edge = False
    for feature in record.features:
        if feature.type == "protocluster" and "True" in feature.qualifiers["contig_edge"]:
            on_edge = True
        break
    if on_edge:
        continue
    token_list = readAntismash5(record.features)
    outfile = open(outdir2 +  genome_name + "_" + f[f.rfind("/")+1:f.rfind(".")],'w')
    i = 0
    for t in token_list:
        if i == 0:
            outfile.write(t)
        else:
            outfile.write(","+t)
        i += 1
    outfile.close()

