# -*- coding: utf-8 -*-
"""
run_regression.py (Refactored)

Consolidates regressionTask, regressionTaskFromScratch, and regressionTaskTest.
"""

import os
import torch
import math
import numpy as np
from torch.utils.data import DataLoader, random_split
from sys import argv

import tools.BGC_MLM_tools as BGC_MLM_tools
from src.utils import arg_parse
from src.data.loading import (load_token_list, parse_regression_file, 
                              load_bgc_tokens)
from src.utils.metrics import evaluate_regression

def main():
    args = arg_parse.parse_args("run_regression", argv[1:])
    
    # Set seed
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    token_list = load_token_list(args.unknown_threshold, args.max_bgc_length)
    
    regression_types, bgc_values = parse_regression_file(args.regression_file)
        
    # Load BGC tokens
    # Using keys of bgc_values as known filter
    bgc_tokens, bgc_names = load_bgc_tokens(
        args.data_set, 
        args.token_path, 
        token_list, 
        args.max_bgc_length,
        known_bgcs=bgc_values
    )
    
    y_vals = []
    for name in bgc_names:
        y_vals.append(bgc_values[name])
        
    dataset = BGC_MLM_tools.BGCRegressionDatasets(
        bgc_tokens, token_list, y_vals, seq_len=args.max_bgc_length
    )
    
    # 2. Setup Splits
    if args.evaluate_only:
        test_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda")
        )
    else:
        # Default split 90/10 hardcoded in original script?
        # Original: math.floor(9*len/10).
        # We can use args.train_fraction if available, or default to 0.9.
        # regressionTask argument map doesn't accept train_fraction though!
        # It accepts 'model_name', 'unknown_threshold'...
        # Wait, I added run_regression args. I didn't add train_fraction?
        # Checking arg_parse.py diff... 
        # I did not add train_fraction to run_regression list in my recent edit.
        # I should assume 0.9 or add it? 
        # I'll stick to 0.9 default/hardcoded to match original behavior if arg not present.
        
        train_fraction = 0.9
        val_size = len(bgc_names) - math.floor(train_fraction * len(bgc_names))
        train_size = len(bgc_names) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda")
        )

    # 3. Model Initialization
    print("Initializing model...")
    bgc_mlm_model = BGC_MLM_tools.BGC_MLM(
        vocab_size=len(token_list),
        seq_len=args.max_bgc_length,
        d_model=args.d_model,
        n_layers=args.n_layers,
        heads=args.heads,
        dropout=args.dropout
    )
    
    mlm = BGC_MLM_tools.MLM(bgc_mlm_model, len(token_list))
    
    freeze = (args.freeze != 0)

    regression_model = BGC_MLM_tools.BGCRegression(
        mlm.bgc_mlm,
        args.d_model,
        len(y_vals[0]),
        freeze=freeze
    )
    
    if args.evaluate_only:
        print(f"Loading regression model weights from {args.model_name}")
        regression_model.load_state_dict(torch.load(args.model_name, map_location=device))
    elif not args.from_scratch:
        print(f"Loading pretrained MLM weights from {args.model_name}")
        mlm.load_state_dict(torch.load(args.model_name, map_location=device))
    else:
         print("Training from scratch")

    regression_model.to(device)

    # 4. Train or Evaluate
    val_dataset_for_trainer = dataset if args.evaluate_only else val_set
    train_loader_for_trainer = None if args.evaluate_only else train_loader
    
    bert_trainer = BGC_MLM_tools.BGCRegressionTrainier(
        regression_model, 
        train_loader_for_trainer, 
        val_dataset_for_trainer, 
        device=device
    )
    
    if not args.evaluate_only:
        print("Starting training...")
        epochs = 400 # Hardcoded in original script? 
        # original regressionTask.py: epochs = 400.
        # I should probably expose it or use args.epochs (default 50 in arg_parse, but regression didn't use it?)
        # My arg_parse update for run_regression DOES NOT include epochs?
        # Checking... I didn't add epochs to run_regression args.
        # I'll stick to 400.
        
        for epoch in range(epochs):
            print(f"Starting epoch: {epoch}")
            bert_trainer.train(epoch)
            
        print("Saving model...")
        final_save_path = args.model_name + "_" + args.model_output if not args.from_scratch else args.model_output
        torch.save(regression_model.state_dict(), final_save_path)
        print(f"Model saved to {final_save_path}")

    # 5. Evaluate
    print("Running evaluation...")
    target_set = dataset if args.evaluate_only else val_set
    predictions = bert_trainer.predict(target_set, batch_size=args.batch_size)
    eval_loader = DataLoader(
        target_set, batch_size=1, shuffle=False, pin_memory=False
    )
    
    evaluate_regression(
        predictions, 
        eval_loader, 
        regression_types, 
        args.model_name,
        plot_dir="."
    )

if __name__ == "__main__":
    main()
