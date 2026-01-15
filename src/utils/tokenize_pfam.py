# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:20:28 2025

@author: Allison Walker
"""

import os
from Bio import SeqIO
#outdir = "tokenized_antismash_mibig4_bacteria/pfam/"
#outdir = "tokenized_genomes_for_ranking/pfam/"
outdir = "tokenized_antismash5_mibig4_all/pfam/"
#in_dir = "genomes_for_bgc_rank_test/"
#in_dir = "antismash5_mibig4_bacteria/"
in_dir = "antismash5_mibig_not_bacteria/"

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
            if (".gbk" in f or ".gb" in f or ".gbff" in f) and "region" not in f:
                fullpath = os.path.join(path, f)
                genome_name = fullpath[0:fullpath.rfind("/")]
                genome_name = genome_name[genome_name.rfind("/")+1:len(genome_name)]
                #check if output exists
                if not os.path.exists(outdir + genome_name + "_" +  f[f.rfind("/")+1:f.rfind(".")]):
                    if ".gbff" in genome_name:
                        if not os.path.exists(outdir + genome_name.replace(".gbff","") + "_" +  f[f.rfind("/")+1:f.rfind(".")]):
                            files.append((genome_name, os.path.join(path, f)))
                    else:
                        if not os.path.exists(outdir + genome_name.replace("genomic","genomic.gbff") + "_" +  f[f.rfind("/")+1:f.rfind(".")]):
                            files.append((genome_name, os.path.join(path, f)))
            #print files
        else:
            if "genometools" in f or "glimmer" in f or "slurm" in f:
                continue
            files = getAllClusterListsInDirectory(os.path.join(path, f), files)
    return files





files = getAllClusterListsInDirectory(in_dir, [])
print(len(files))
for genome_name, f in files:
    print(f)
    record = SeqIO.read(open(f, 'rU'),"genbank")
    #check if cluster is on the contig edge
    
    token_list = readAntismash5(record.features)
    #outfile = open(outdir +  genome_name + "_" + f[f.rfind("/")+1:f.rfind(".")],'w')
    outfile = open(outdir +  genome_name,'w')
    i = 0
    for t in token_list:
        if i == 0:
            outfile.write(t)
        else:
            outfile.write(","+t)
        i += 1
    outfile.close()
