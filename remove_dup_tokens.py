# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:02:52 2024

@author: allis
"""
import os
import subprocess

indir = "tokenized_bgcs/pfam2/"

for f in os.listdir(indir):
    if ".gbff" not in f:
        other_option = f.replace("genomic", "genomic.gbff")
        if os.path.exists(indir+other_option):
            subprocess.call(["rm",indir+other_option])