# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset, DataLoader

import sqlite3

def load_token_list(db_path, unknown_threshold):
    """
    Loads the token list from the SQLite database filtering by prevalence threshold.
    """
    token_list = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query tokens that meet or exceed the unknown_threshold
        cursor.execute("SELECT PFAM FROM TokenList WHERE prevalence >= ?", (unknown_threshold,))
        
        for row in cursor.fetchall():
            token_list.append(row[0])
            
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to load tokens from {db_path} - {e}")
        
    return token_list

def parse_classification_file(classification_file_path):
    """
    Parses the BGC classification CSV file.
    Returns:
        classification_types (list): List of class names or indices.
        classification_counts (dict): Count of positives for each class.
        bgc_classifications (dict): Mapping of bgc_name -> list of int labels.
        total_count (int): Total count of positive labels.
    """
    classification_types = []
    classification_counts = {}
    bgc_classifications = {}
    total_count = 0
    
    with open(classification_file_path, 'r') as classification_file:
        for line in classification_file:
            split_line = line.replace("\n","").split(",")
            if "bgc_id" in line:
                for t in split_line[1:len(split_line)]:
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
                if bgc_name in bgc_classifications:
                    continue
                bgc_classifications[bgc_name] = []
                i = 0
                for val in split_line[1:len(split_line)]:
                    bgc_classifications[bgc_name].append(int(val))
                    if int(val) == 1:
                        classification_counts[classification_types[i]] += 1
                        total_count += 1
                    i += 1
    return classification_types, classification_counts, bgc_classifications, total_count

def parse_regression_file(regression_file_path):
    """
    Parses the regression value CSV file.
    Returns:
        regression_types (list): List of regression target names.
        bgc_values (dict): Mapping of bgc_name -> list of float values.
    """
    regression_types = []
    bgc_values = {}
    
    with open(regression_file_path, 'r') as regression_file:
        for line in regression_file:
            split_line = line.replace("\n","").split(",")
            if "bgc_id" in line:
                for t in split_line[1:len(split_line)]:
                   if t not in regression_types:
                       regression_types.append(t)
            else:
                bgc_name = split_line[0]
                if bgc_name in bgc_values:
                    continue
                bgc_values[bgc_name] = []
                for val in split_line[1:len(split_line)]:
                    bgc_values[bgc_name].append(float(val))
    return regression_types, bgc_values

def parse_fp_file(fp_file_path):
    """
    Parses the fingerprint (metric learning) CSV file.
    Returns:
        classification_types (list): List of FP indices/names (often just headers).
        fps (dict): Mapping of bgc_name -> list of int (fingerprint bits).
    """
    classification_types = []
    fps = {}
    
    with open(fp_file_path, 'r') as fp_file:
        for line in fp_file:
            split_line = line.replace("\n","").split(",")
            if "bgc_id" in line:
                for t in split_line[1:len(split_line)]:
                   if t not in classification_types:
                       classification_types.append(t)
            else:
                bgc_name = split_line[0]
                if len(classification_types) == 0:
                    #if no header then number classifications
                    for i in range(1,len(split_line)):
                        classification_types.append(i-1)    
                if bgc_name in fps:
                    continue
                fps[bgc_name] = []
                for val in split_line[1:len(split_line)]:
                    fps[bgc_name].append(int(val))
    return classification_types, fps

def load_bgc_tokens(db_path, token_list, max_bgc_length, known_bgcs=None):
    """
    Loads BGC tokens from SQLite db.
    Args:
        db_path (str): Path to the SQLite database.
        token_list (list): Valid tokens list.
        max_bgc_length (int): Max length.
        known_bgcs (dict/set): Optional set of BGC names to filter by (e.g. only those with labels).
                               Matches against refseq_assembly or locus.
    
    Returns:
        bgc_tokens (list): List of token lists.
        bgc_names (list): List of BGC names corresponding to tokens.
    """
    bgc_tokens = []
    bgc_names = []
    
    token_set = set(token_list)  # O(1) lookups
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, refseq_assembly, locus, region, tokenized_sequence FROM RegionSequences")
        
        for row in cursor.fetchall():
            row_id, refseq, locus, region, tokenized_seq = row
            
            # Form possible matching names
            possible_names = [str(row_id), refseq, locus, f"{refseq}_region{region:03d}", f"{locus}_region{region:03d}"]
            
            bgc_name = None
            if known_bgcs is not None:
                for name in possible_names:
                    if name in known_bgcs:
                        bgc_name = name
                        break
                if bgc_name is None:
                    continue # Skip if not found
            else:
                # Default to a descriptive BGC name if no filter provided
                bgc_name = f"{refseq}_{locus}_region{region}"

            if not tokenized_seq:
                continue

            # Tokenized sequence in DB is a comma separated string
            split_line = [t.strip() for t in tokenized_seq.split(",")]
            
            if len(split_line) == 0:
                continue
            
            # +2 for CLS and SEP tokens
            if len(split_line) + 2 > max_bgc_length:
                continue
                
            bgc_token = ["CLS"]
            for t in split_line:
                if t in token_set:
                    bgc_token.append(t)
                else:
                    bgc_token.append("UNK")
            bgc_token.append("SEP")
            
            # Vectorized padding logic avoiding loop
            pad_length = max_bgc_length - len(bgc_token)
            if pad_length > 0:
                bgc_token.extend(["PAD"] * pad_length)
                   
            bgc_names.append(bgc_name)           
            bgc_tokens.append(bgc_token)
            
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to load sequence data from {db_path} - {e}")
        
    print(str(len(bgc_tokens)) + " BGCs under length threshold")
    return bgc_tokens, bgc_names
