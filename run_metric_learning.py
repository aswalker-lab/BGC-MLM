# -*- coding: utf-8 -*-
"""
run_metric_learning.py (Refactored)

Consolidates metricTask and metricTaskFromScratch.
"""

import os
import torch
import math
import numpy as np
from torch.utils.data import DataLoader, random_split
from sys import argv

import tools.BGC_MLM_tools as BGC_MLM_tools
from src.utils import arg_parse
from src.data.loading import (load_token_list, parse_fp_file, 
                              load_bgc_tokens)

def main():
    args = arg_parse.parse_args("run_metric_learning", argv[1:])
    
    # Set seed
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    token_list = load_token_list(args.unknown_threshold, args.max_bgc_length)
    
    classification_types, fps = parse_fp_file(args.fp_file)
        
    # Load BGC tokens
    bgc_tokens, bgc_names = load_bgc_tokens(
        args.data_set, 
        args.token_path, 
        token_list, 
        args.max_bgc_length,
        known_bgcs=fps
    )
    
    target_embedding = []
    for name in bgc_names:
        target_embedding.append(fps[name])
        
    dataset = BGC_MLM_tools.BGCTargetEmbeddingDataset(
        bgc_tokens, token_list, target_embedding, seq_len=args.max_bgc_length
    )
    
    # 2. Setup Splits
    if args.evaluate_only:
         # Metric learning usually outputs embeddings, evaluation might be similarity check?
         # metricTask.py executes training loops.
         # There is no metricTaskTest.py?
         # "metricTaskFromScratch.py" exists.
         # If evaluate_only, maybe we just predict/save embeddings?
         # Original script ends with saving model.
         # I'll support evaluate_only by just running predict?
         test_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda")
         )
    else:
        val_size = len(bgc_names) - math.floor(args.train_fraction * len(bgc_names))
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
    
    # Note: metricTask doesn't use freeze arg? 
    # BGCMetricLearning allows freeze arg in init?
    # Checking tools/BGC_MLM_tools.py BGCMetricLearning:
    # def __init__(self, bgc_mlm, d_model, d_fp, freeze=False):
    # original script: target_embedding_model = BGC_MLM_tools.BGCMetricLearning(mlm.bgc_mlm,args.d_model,args.fp_size)
    # So it uses default freeze=False.
    
    metric_model = BGC_MLM_tools.BGCMetricLearning(
        mlm.bgc_mlm,
        args.d_model,
        args.fp_size,
        freeze=False 
    )
    
    if args.evaluate_only:
        print(f"Loading metric model weights from {args.model_name}")
        # Assuming model_name points to the BGCMetricLearning state dict
        metric_model.load_state_dict(torch.load(args.model_name, map_location=device))
    elif not args.from_scratch:
        print(f"Loading pretrained MLM weights from {args.model_name}")
        mlm.load_state_dict(torch.load(args.model_name, map_location=device))
    else:
        print("Training from scratch")

    metric_model.to(device)

    # 4. Train or Evaluate
    val_dataset_for_trainer = dataset if args.evaluate_only else val_set
    train_loader_for_trainer = None if args.evaluate_only else train_loader
    
    bert_trainer = BGC_MLM_tools.BGCMetricTrainer(
        metric_model, 
        train_loader_for_trainer, 
        val_dataset_for_trainer, 
        device=device,
        loss_type=args.loss_type
    )
    
    if not args.evaluate_only:
        print("Starting training...")
        epochs = args.epochs
        for epoch in range(epochs):
            print(f"Starting epoch: {epoch}")
            bert_trainer.train(epoch)
            
        print("Saving model...")
        final_save_path = args.model_name + "_" + args.model_output if not args.from_scratch else args.model_output
        torch.save(metric_model.state_dict(), final_save_path)
        print(f"Model saved to {final_save_path}")

    # No specific evaluation metric output in original script besides loss during training.
    # predict() exists.
    if args.evaluate_only:
        print("Running prediction (evaluation)...")
        predictions = bert_trainer.predict(dataset, batch_size=args.batch_size)
        # Maybe save predictions?
        # For now, just print shape
        print(f"Predictions shape: {predictions.shape}")

if __name__ == "__main__":
    main()
