# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 2025

@author: Allison Walker (Refactored)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             precision_score, recall_score, roc_auc_score, 
                             average_precision_score, mean_squared_error, 
                             mean_absolute_error)
from tqdm import tqdm

def evaluate_classification(predictions, data_loader, classification_types, model_name, outfile_path=None):
    """
    Evaluates classification predictions against ground truth from the data_loader.
    
    Args:
        predictions (torch.Tensor): Model predictions (logits or probabilities). 
                                     Trainer.predict returns them, usually logits? 
                                     Wait, BGCMultiLabelTrainer.predict applies Sigmoid!
                                     So these are probabilities.
        data_loader (DataLoader): DataLoader acting as the Truth source.
        classification_types (list): List of class names.
        model_name (str): Name of model for logging.
        outfile_path (str): Path to CSV/Text file to append metrics.
    """
    
    # Prepare output file if needed
    outfile = None
    if outfile_path:
        # Check if file exists to write header
        write_header = not os.path.exists(outfile_path)
        outfile = open(outfile_path, 'a')
        if write_header:
            outfile.write("model_name,accuracy,balanced_accuracy,precision,recall,ROC_AUC,PRC_AUC\n")

    print(f"Predictions shape: {predictions.shape}")

    for i in range(len(classification_types)):
        task_name = str(classification_types[i])
        print(task_name)
        
        # Extract true labels for this specific task/class index i
        true_y = []
        # We need to iterate over the loader again to get labels aligned?
        # Assuming data_loader is not shuffled or is the same one used for prediction.
        # The trainer.predict iterates the loader. 
        # If the loader was shuffled, we might have issues if we don't ensure alignment.
        # But usually predict and this check use the same loader instance (shuffle=False).
        
        # Note: BGCClassificationDatasets returns 'classification_label'
        for j, data in enumerate(data_loader):
            # data['classification_label'] shape is (batch, num_classes)
            # We want the i-th class for each sample
            labels = data["classification_label"].detach().cpu().numpy()
            for label_row in labels:
                true_y.append(int(label_row[i]))
        
        if sum(true_y) < 2:
            print("not enough of class")
            continue
            
        # Get predicted probabilities for this class
        predicted_y_probs = predictions.detach().cpu().numpy()[:, i]
        
        # Threshold at 0.5 for binary metrics
        predicted_y_class = [x > 0.5 for x in predicted_y_probs]
        
        try:
            accuracy = accuracy_score(true_y, predicted_y_class)
            balanced_accuracy = balanced_accuracy_score(true_y, predicted_y_class)
            precision = precision_score(true_y, predicted_y_class, zero_division=0)
            recall = recall_score(true_y, predicted_y_class, zero_division=0)
            roc_auc = roc_auc_score(true_y, predicted_y_probs)
            # Try/catch for PRC AUC if strict? usually okay. 
            pass
        except ValueError as e:
            print(f"Error calculating metrics for {task_name}: {e}")
            continue

        # classificationTaskTest.py uses average_precision_score for PRC_AUC
        prc_auc = average_precision_score(true_y, predicted_y_probs)

        print("accuracy: " + str(accuracy))
        print("balanced accuracy: " + str(balanced_accuracy))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("ROC AUC: " + str(roc_auc))
        
        if outfile:
            # model_name processing to get basename if path
            short_model_name = os.path.basename(model_name)
            outfile.write(f"{short_model_name}_{task_name},{accuracy},{balanced_accuracy},{precision},{recall},{roc_auc},{prc_auc}\n")

    if outfile:
        outfile.close()

def evaluate_regression(predictions, data_loader, regression_types, model_name, plot_dir="."):
    """
    Evaluates regression predictions.
    """
    
    print(f"Predictions shape: {predictions.shape}")

    for i in range(len(regression_types)):
        task_name = str(regression_types[i])
        print(task_name)
        
        true_y = []
        for j, data in enumerate(data_loader):
            vals = data["regression_value"].detach().cpu().numpy()
            for val_row in vals:
                true_y.append(float(val_row[i])) # ensure float
        
        predicted_y = predictions.detach().cpu().numpy()[:, i]
        
        mse = mean_squared_error(true_y, predicted_y)
        mae = mean_absolute_error(true_y, predicted_y)
        
        # Correlation
        corr, _ = pearsonr(true_y, predicted_y)
        
        print("MSE: " + str(mse))
        print("MAE: " + str(mae))
        print("Pearson's correlation: " + str(corr))
        
        # Plotting
        try:
            plt.figure()
            min_val = min(min(true_y), min(predicted_y))
            max_val = max(max(true_y), max(predicted_y))
            
            # Scatter
            # Using basic color, maybe separate train vs val logic later if needed
            # But here we just evaluate one set
            plt.scatter(true_y, predicted_y, c='blue')
            
            # Limits (add a bit of padding if needed, or just strict min/max as in original)
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            
            plt.xlabel("True value")
            plt.ylabel("Predicted value")
            plt.title(task_name)
            
            save_path = os.path.join(plot_dir, f"{task_name}.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error plotting {task_name}: {e}")

