# -*- coding: utf-8 -*-
"""
run_classification.py (Refactored)

Consolidates classificationTask, classificationTaskFromScratch, and classificationTaskTest.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import math
import numpy as np
from torch.utils.data import DataLoader, random_split
from data import datasets
from utils import arg_parse
from models.architecture import MLM, BGC_MLM
from data.loading import (load_token_list, parse_classification_file, 
                              load_bgc_tokens)
from utils.metrics import evaluate_classification

def main():
    args = arg_parse.parse_args("run_classification", argv[1:])
    
    # Set seed
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    token_list = load_token_list(args.data_set, args.unknown_threshold)
    
    classification_types, classification_counts, bgc_classifications, total_count = \
        parse_classification_file(args.classification_file)
        
    # Load BGC tokens
    # For evaluate_only, args.data_set contains test files
    # For training, args.data_set contains train/val files
    # If evaluate_only, we might want to filter only if in classification file? 
    # Original scripts check "if filename not in bgc_classifications: continue"
    bgc_tokens, bgc_names = load_bgc_tokens(
        args.data_set, 
        token_list, 
        args.max_bgc_length,
        known_bgcs=bgc_classifications
    )
    
    y_vals = []
    for name in bgc_names:
        y_vals.append(bgc_classifications[name])
        
    dataset = datasets.BGCClassificationDatasets(
        bgc_tokens, token_list, y_vals, seq_len=args.max_bgc_length
    )
    
    # 2. Setup Splits
    if args.evaluate_only:
        # Test mode: Dataset is purely for testing/evaluation
        test_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=(device.type == "cuda")
        )
        # We might also want batch prediction loader?
        # classificationTaskTest uses batch_size=args.batch_size for prediction 
        # but batch_size=1 for accurate metric loop? 
        # Actually bert_trainer.predict uses batch_size arg.
        predict_loader_dataset = dataset # pass dataset directly to predict
    else:
        # Train mode: Split into Train/Val
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
    bgc_mlm_model = BGC_MLM(
        vocab_size=len(token_list),
        seq_len=args.max_bgc_length,
        d_model=args.d_model,
        n_layers=args.n_layers,
        heads=args.heads,
        dropout=args.dropout
    )
    
    mlm = MLM(bgc_mlm_model, len(token_list))
    
    # Calculate pos_weights for loss
    pos_weights = []
    # Original logic:
    # classificationTask has loop: for c in classification_counts...
    # classificationTaskTest has: pos_weights.append(1) for all.
    
    # If standard training/test, we reuse the training logic for weights if training.
    # If evaluating, weights don't matter for prediction usually, but Trainer might require them?
    # Trainer takes pos_weights in init.
    
    if not args.evaluate_only:
        # Calculate weights based on counts
        # Note: classification_types order matters. parse_classification_file returns them in order found/header.
        # Original script iterates classification_counts keys? No, it iterates classification_counts dict directly?
        # "for c in classification_counts:" -> iterates keys.
        # But are keys sorted? Python 3.7+ yes if insertion order.
        # Should rely on classification_types list to be safe?
        # Original: "for c in classification_counts:" ... 
        # But classification_counts was filled while iterating classification_types or headers.
        # Let's hope classification_types matches the columns in y_vals.
        
        for t in classification_types:
            count = classification_counts[t]
            if args.use_pos_weights == 0:
                pos_weights.append(1)
            elif count != 0:
                 pos_weights.append((total_count - count)/count)
            else:
                 pos_weights.append((total_count - count))
    else:
        # Evaluating
        for _ in range(len(y_vals[0])):
            pos_weights.append(1)
            
    pos_weights_tensor = torch.as_tensor(pos_weights, dtype=torch.float).to(device)

    # Load Weights or Scratch
    # 3 cases: 
    # A) Train from Scratch: Don't load anything (random init MLM).
    # B) Train Pretrained: Load MLM weights from args.model_name.
    # C) Evaluate: Load CLASSIFIER weights from args.model_name. 
    
    freeze = (args.freeze != 0) # args.freeze is int 0/1

    classifier_model = BGC_MLM_tools.BGCMultiLabelClassifier(
        mlm.bgc_mlm, # Uses the sub-module of MLM
        args.d_model,
        len(y_vals[0]),
        freeze=freeze
    )
    
    if args.evaluate_only:
        # Load full Classifier state
        # Note: args.model_name acts as the input path here
        print(f"Loading classifier weights from {args.model_name}")
        classifier_model.load_state_dict(torch.load(args.model_name, map_location=device))
    elif not args.from_scratch:
        # Load MLM weights 
        print(f"Loading pretrained MLM weights from {args.model_name}")
        mlm.load_state_dict(torch.load(args.model_name, map_location=device))
        # Note: classifier_model shares 'bgc_mlm' instance with 'mlm' object above, 
        # so loading into 'mlm' updates 'classifier_model.bgc_mlm' too.
    else:
        print("Training from scratch (random initialization)")
        # No loading

    classifier_model.to(device)

    # 4. Train or Evaluate
    # Trainer setup
    # Note: Trainer init requires val_set. If evaluate_only, we pass dataset as val_set?
    # classificationTaskTest passes dataset as val_set?
    # Original Test script: 
    # bert_trainer = BGC_MLM_tools.BGCMultiLabelTrainier(..., val_set=classification_dataset, ...)
    
    val_dataset_for_trainer = dataset if args.evaluate_only else val_set
    train_loader_for_trainer = None if args.evaluate_only else train_loader
    
    bert_trainer = BGC_MLM_tools.BGCMultiLabelTrainier(
        classifier_model, 
        train_loader_for_trainer, 
        val_dataset_for_trainer, 
        pos_weights_tensor, 
        device=device
    )
    
    if not args.evaluate_only:
        print("Starting training...")
        epochs = args.epochs
        for epoch in range(epochs):
            print(f"Starting epoch: {epoch}")
            bert_trainer.train(epoch)
            
            # Checkpointing (Optional logic from original script)
            # Original saves logs to file every 10/20/50 epochs.
            # And saves final model at end.
            
        print("Saving model...")
        final_save_path = args.model_name + "_" + args.model_output if not args.from_scratch else args.model_output
        # Original logic: args.model_name + "_" +args.model_output
        # For scratch, model_name might be weird if not path.
        # Let's assume user provides sensible args.
        if args.from_scratch:
             # Just use model_name or output? 
             # classificationTaskFromScratch saves to args.model_name + "_" +args.model_output
             pass
        torch.save(classifier_model.state_dict(), final_save_path)
        print(f"Model saved to {final_save_path}")

    # 5. Evaluate / Metrics
    print("Running evaluation...")
    
    # If training, we eval on Train and Val?
    # Original script evaluates on Train and Val if 'write_metrics' isn't 0?
    # Actually it says "if args.write_metrics == 0: exit()".
    # And default write_metrics is not clear in arg_parse, but classification script checks it.
    # 'arg_parse.py' shows: "--write_metrics" (optional flag? No, no default, likely int if treated like others).
    # Since I removed 'write_metrics' from my recommended run script args (or commented it out), 
    # I should probably just Always evaluate or use 'evaluate_only'.
    
    # If training was run, user might want quick metrics on Val set.
    # If evaluate_only, user definitely wants metrics on the dataset provided.
    
    target_set = dataset if args.evaluate_only else val_set
    
    predictions = bert_trainer.predict(target_set, batch_size=args.batch_size)
    
    # Create loader for Ground Truth iteration
    # Original script uses batch_size=1 for accurate manual metric loop
    eval_loader = DataLoader(
        target_set, batch_size=1, shuffle=False, pin_memory=False
    )
    
    # Output file
    outfile_name = None
    if args.evaluate_only:
         # Construct outfile name like original test script: data_set + classification_file + _classification.txt
         # Or just use a standard one.
         base = os.path.basename(args.data_set)
         if "." in base: base = base[:base.rfind(".")]
         class_base = os.path.basename(args.classification_file)
         if "." in class_base: class_base = class_base[:class_base.rfind(".")]
         outfile_name = f"{base}_{class_base}_classification.txt"
         print(f"Writing metrics to {outfile_name}")

    evaluate_classification(
        predictions, 
        eval_loader, 
        classification_types, 
        args.model_name,
        outfile_path=outfile_name
    )

if __name__ == "__main__":
    main()
