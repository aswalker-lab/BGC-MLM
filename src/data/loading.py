# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 2025

@author: Allison Walker (Refactored)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

def load_token_list(unknown_threshold, max_bgc_length):
    """
    Loads the token list from a file named based on threshold and length.
    """
    token_list = []
    filename = "token_list_" + str(unknown_threshold) + "_" + str(max_bgc_length) + ".txt"
    try:
        token_list_file = open(filename)
        for line in token_list_file:
            token_list.append(line.replace("\n",""))
        token_list_file.close()
    except FileNotFoundError:
        # Just return empty or re-raise with message? Script usually crashes if this fails.
        # Returning empty to be handled by caller or crash later.
        print(f"Warning: {filename} not found.")
        
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

def load_bgc_tokens(data_set_file, token_path, token_list, max_bgc_length, known_bgcs=None):
    """
    Loads BGC tokens for a list of files specified in data_set_file.
    Args:
        data_set_file (str): Path to file containing list of BGC filenames.
        token_path (str): Directory containing token files.
        token_list (list): Valid tokens list.
        max_bgc_length (int): Max length.
        known_bgcs (dict/set): Optional set of BGC names to filter by (e.g. only those with labels).
    
    Returns:
        bgc_tokens (list): List of token lists.
        bgc_names (list): List of BGC names corresponding to tokens.
    """
    bgc_tokens = []
    bgc_names = []
    
    with open(data_set_file, 'r') as training_data_file:
        for line in training_data_file:
            filename = line.replace("\n","")
            # If known_bgcs is provided, skip if not in it
            if known_bgcs is not None and filename not in known_bgcs:
                continue
            
            bgc_name = filename
            
            # Construct full path to token file
            # Assuming token_path + filename is the pattern (classificationTask.py line 98)
            full_path = os.path.join(token_path, filename)
            
            if not os.path.exists(full_path):
                # Fallback or strict error? Original code just tries to open it.
                # skipping if not found to avoid crash
                continue

            infile = open(full_path)
            bgc_token = ["CLS"]
            # useBGC = True # unused in original script
            
            # Original code logic:
            # for line in infile: split_line = line.split(",")
            # if len(split_line) + 2 > max_bgc_length: continue
            
            # We need to read the content. Assuming one line per file or just concatenation?
            # Original script loop:
            # for line in infile:
            #    split_line = line.split(",")
            # It seems it overwrites split_line or just takes the last one? 
            # Likely the file is a single line CSV of tokens.
            
            split_line = []
            for line_in_file in infile:
                split_line = line_in_file.split(",")
            infile.close()
            
            if len(split_line) == 0:
                continue

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
                   
            bgc_names.append(bgc_name)           
            bgc_tokens.append(bgc_token)
            
    print(str(len(bgc_tokens)) + " BGCs under length threshold")
    return bgc_tokens, bgc_names
