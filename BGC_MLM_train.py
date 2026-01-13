# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 09:35:51 2025

@author: Allison Walker

To Learn:
    - How are the tokens created?
    - How is the data formatted?
    - Why would there be unknown tokens?
    - Why are we dynamically creating the tokens, I thought the tokens were being created in other scripts? 
        Are we doubling up on the token creation process?  

To Do:
    - Clean up data loading/parsing process, definitely a more efficient method possible depending on data input
    - Setup a debug mode to toggle a lot of these terminal outputs and extra tracking vars
    - 

"""

# adapted from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

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
import tools.default_parse
import tools.BGC_MLM_tools
from sys import argv


class BGCTrainer:
    """
    Class for training a BGC Masked Language Model (MLM).
    Encapsulates data loading, model initialization, training, and saving.
    """

    def __init__(self, args_dict):
        """
        Initialize the trainer by parsing arguments and setting up the environment.

        Args:
            args_dict (dict): Dictionary of arguments.
        """
        
        # Output input arguments
        self.args_dict = args_dict
        print("Arguments:")
        for k, v in self.args_dict.items():
            print(f"{k}: {v}")

        # Set seed for reproducibility
        torch.manual_seed(self.args_dict["seed"])

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set thread count
        torch.set_num_threads(self.args_dict['thread_count'])

        # Store config variables
        self.unknown_threshold = self.args_dict["unknown_threshold"]
        self.max_bgc_length = self.args_dict["max_bgc_length"]
        self.training_datasets_file = self.args_dict["training_datasets_file"]
        self.model_name = self.args_dict["model_name"]
        
        # Initialize state variables
        self.bgc_tokens = None
        self.padded_bgcs = None
        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.bgc_mlm_model = None
        self.mlm = None
        self.mlm_trainer = None

        self.token_list = ["PAD", "CLS", "SEP", "MASK", "UNK"]
        """
        Token Meanings:
            PAD: Padding token so all inputs are the same length
            CLS: Classification token, start of sentence (SOS)
            SEP: Separator token, end of sentence (EOS)
            MASK: Masked token, used for masked language modeling
            UNK: Unknown token, used for tokens outside tokanizer vocabulary
        """

    def load_data(self):
        """
        Loads and processes the BGC data.
        - Reads files listed in the training data file.
        - Tokenizes BGCs.
        - Filters based on length.
        - Handles unknown tokens based on frequency threshold.
        - Pads sequences.
        - Creates the dataset and train/val splits.
        """

        print("Loading tokens...")
        with open(self.training_datasets_file) as file:
            training_data_files = file.readlines()
        bgc_tokens = []
        token_counts = {"CLS": 0, "SEP": 0}

        # First pass: read all tokens and count frequencies
        """
        I am going to assume the data structure is as follows:
            - training dataset file: each line is a unique path to a subset of the data
            - data files: each line is a unique BGC as a list of tokens seperated with commas
        """
        for path in training_data_files:           
            try:
                with open(path) as infile:
                    current_bgc_token = ["CLS"]
                    
                    for line in infile:
                        split_line = line.strip().split(",")
                        # Check length constraint
                        if len(split_line) + 2 > self.max_bgc_length:
                            break
                        
                        for t in split_line:
                            current_bgc_token.append(t)
                            if t not in token_counts:
                                token_counts[t] = 0
                            token_counts[t] += 1


                    current_bgc_token.append("SEP")
                    token_counts["CLS"] += 1
                    token_counts["SEP"] += 1
                    bgc_tokens.append(current_bgc_token)

            except FileNotFoundError:
                print(f"Warning: File not found: {path}")
                continue


        print(f"{len(bgc_tokens)} BGCs under length threshold")

        # Second pass: Replace rare tokens with UNK and build final vocab
        print("Processing tokens and handling unknowns...")
        self.bgc_tokens = []
        for token in bgc_tokens:
            new_bgc_token = []
            for t in token:
                if t in token_counts and token_counts[t] < self.unknown_threshold:
                    new_bgc_token.append("UNK")
                else:
                    new_bgc_token.append(t)
                    if t not in self.token_list:
                        self.token_list.append(t)
            self.bgc_tokens.append(new_bgc_token)

        print(f"{len(self.token_list)} tokens past count threshold")

        # Third pass: Pad BGCs
        print("Padding BGCs...")
        self.padded_bgcs = []
        for bgc in self.bgc_tokens:
            while len(bgc) < self.max_bgc_length:
                bgc.append("PAD")
            self.padded_bgcs.append(bgc)

        # Save token list
        token_filename = f"token_list_{self.unknown_threshold}_{self.max_bgc_length}.txt"
        with open(token_filename, 'w') as outfile:
            for t in self.token_list:
                outfile.write(t + "\n")

        # Create Dataset
        self.dataset = tools.BGC_MLM_tools.MLMDataset(
            self.padded_bgcs, self.token_list, seq_len=self.max_bgc_length
        )

        # Split Train/Val
        total_len = len(self.padded_bgcs)
        train_len = math.floor(9 * total_len / 10)
        val_len = total_len - train_len
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [train_len, val_len]
        )

        print(f"Train set length: {len(self.train_set)}")
        print(f"Val set length: {len(self.val_set)}")
        print(f"Vocab size: {len(self.token_list)}")
        if len(self.padded_bgcs) > 0:
            print(f"Sample padded BGC: {self.padded_bgcs[0]}")

    def init_model(self):
        """
        Initializes the BGC MLM model and the underlying MLM wrapper.
        """
        if self.token_list is None:
            raise ValueError("Token list is not initialized. Run load_data() first.")

        # Initialize base model
        self.bgc_mlm_model = tools.BGC_MLM_tools.BGC_MLM(
            vocab_size=len(self.token_list),
            seq_len=self.max_bgc_length,
            d_model=self.args_dict["d_model"],
            n_layers=self.args_dict["n_layers"],
            heads=self.args_dict["heads"],
            dropout=self.args_dict["dropout"]
        )

        # Initialize MLM wrapper
        self.mlm = tools.BGC_MLM_tools.MLM(self.bgc_mlm_model, len(self.token_list))
        self.mlm.to(self.device)

        # Print parameters and buffers
        for name, param in self.mlm.named_parameters():
            print(name)
            print(param.device)
        
        for name, buffer in self.mlm.named_buffers():
            print(name)
            print(buffer.device)

    def train(self):
        """
        Runs the training loop.
        """
        if self.train_set is None or self.mlm is None:
             raise ValueError("Data or Model not initialized.")

        # Create DataLoader
        train_loader = DataLoader(
            self.train_set, 
            batch_size=self.args_dict["batch_size"], 
            shuffle=True, 
            pin_memory=True
        )

        # Initialize Trainer
        self.mlm_trainer = tools.BGC_MLM_tools.MLMTrainer(
            self.mlm, train_loader, self.val_set, device=self.device
        )
        
        epochs = 50 

        for epoch in range(epochs):
            print(f"Starting epoch: {epoch}")
            self.mlm_trainer.train(epoch)
            
            # Checkpoints matching original logic
            if epoch == 10:
                self.save_output()
            if epoch == 25:
                self.save_output()
        
        # Final save
        self.save_output()
        
        # Save Model State
        torch.save(self.mlm.state_dict(), self.model_name)
        print(f"Model saved to {self.model_name}")

    def save_output(self):
        """
        Saves the current training and validation loss to a file.
        """
        filename = "loss_file_" + self.model_name + ".txt"
        with open(filename, 'w') as loss_outfile:
            for i in range(len(self.mlm_trainer.train_loss_list)):
                loss_outfile.write(
                    f"{i},{self.mlm_trainer.train_loss_list[i]},{self.mlm_trainer.val_loss_list[i]}\n"
                )
        print(f"Loss history saved to {filename}")


if __name__ == "__main__":
    # Parse command line arguments
    args_dict = tools.default_parse.parse("BGC_MLM_train", argv[1:])
    
    # Create trainer instance
    trainer = BGCTrainer(vars(args_dict))
    
    # Execute pipeline
    trainer.load_data()
    trainer.init_model()
    trainer.train()
