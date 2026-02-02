"""
File to collate and process data for BGC-MLM with the goal of limiting duplicate operations and
overall improving the resource efficiency of the project
"""
import time
import sys
import numpy as np
import csv

def process_data(bgc_file, bgc_classification, bgc_regression, bgc_embeddings, output_dir):
    """
    Processes data for BGC-MLM
    """
    
    output_file_name = sys.path.cat(output_dir, f"processed_data_{time.date()}")

    processed_data = {}
    