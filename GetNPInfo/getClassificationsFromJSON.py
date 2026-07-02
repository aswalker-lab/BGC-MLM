# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:52:11 2025

@author: Allison Walker
"""

import json
import os

classification_list = []
bgc_classifications = {}

for f in os.listdir("mibig_json_4.0"):
    infile = open("mibig_json_4.0/" +f)
    data = json.load(infile)
    biosynthesis_classes = data["biosynthesis"]["classes"]
    bgc_classifications[f[0:f.rfind(".")]] = []
    for c in biosynthesis_classes:
        np_class = c["class"]
        bgc_classifications[f[0:f.rfind(".")]].append(np_class)
        if np_class not in classification_list:
            classification_list.append(np_class)
        
outfile = open("mibig_bacterial_classes.csv",'w')
outfile.write("bgc_id")
for c in classification_list:
    outfile.write("," + c)
outfile.write("\n")
for bgc in bgc_classifications:
    outfile.write(bgc)
    for c in classification_list:
        if c in bgc_classifications[bgc]:
            outfile.write(",1")
        else:
            outfile.write(",0")
    outfile.write("\n")
outfile.close()