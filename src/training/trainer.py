# -*- coding: utf-8 -*-

# adapted from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

import os
import math
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from ..models.factory import (
        pairwise_cosine_vector,
        cosine_similarity_matrix,
        cosine_distance_matrix
    )
from ..models.architecture import BGC_MLM, MLM


class BaseLightningModule(pl.LightningModule):
    """
    Abstract base class for all BGC Lightning modules.
    Handles optimization scheduling and common setup.
    """
    def __init__(
        self,
        model,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        d_model=None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        
        # Try to infer d_model if not provided, for the scheduler
        if d_model is None:
            if hasattr(model, 'd_model'):
                self.d_model = model.d_model
            elif hasattr(model, 'bgc_mlm') and hasattr(model.bgc_mlm, 'd_model'):
                self.d_model = model.bgc_mlm.d_model
            else:
                self.d_model = 256 # Fallback default or raise error?
                print("Warning: d_model not found, using default 256 for scheduler")
        else:
            self.d_model = d_model

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Initialize optimizer
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.d_model**(-0.5), # Initial LR for Custom Schedule
            betas=self.betas, 
            weight_decay=self.weight_decay
        )

        # Custom Scheduler Logic mimicking ScheduledOptim
        def lr_lambda(step):
            # step counts from 0, original code increments first then calculates
            current_step = step + 1 
            if current_step < self.warmup_steps:
                return np.power(self.warmup_steps, -1.7) * current_step
            else:
                return np.min([
                    np.power(current_step, -0.45),
                    np.power(self.warmup_steps, -0.7)
                ])

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
            'name': 'custom_scheduler'
        }
        return [optimizer], [scheduler]


class MLMLightningModule(BaseLightningModule):
    def __init__(self, model, vocab_size=None, **kwargs):
        super().__init__(model, **kwargs)
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.vocab_size = vocab_size

    def training_step(self, batch, batch_idx):
        # batch is a dict
        mask_lm_output = self.model.forward(batch["bert_input"])
        mask_loss = self.criterion(
            mask_lm_output.transpose(1, 2), batch["bert_label"]
        )
        self.log("train_loss", mask_loss, prog_bar=True)
        return mask_loss

    def validation_step(self, batch, batch_idx):
        mask_lm_output = self.model.forward(batch["bert_input"])
        mask_loss = self.criterion(
            mask_lm_output.transpose(1, 2), batch["bert_label"]
        )
        self.log("val_loss", mask_loss, prog_bar=True)
        
        # Calculate Metrics logic from original code
        # Specifically accuracy, top-5, top-10
        # This can be expensive, so maybe do it less frequently or lighter version?
        # Original code did this loop-wise.
        
        # Vectorized implementation of accuracy
        predictions = torch.argmax(mask_lm_output, dim=2) # (B, Seq)
        labels = batch["bert_label"] # (B, Seq)
        
        # Mask special tokens (0=PAD, 1=CLS, 2=SEP typically but original code checks 0, 1, 2)
        # Original: if data["bert_label"][0][predicted_index] in [0,1,2] -> continue
        # The logic was weird: predicted_index = torch.argmax(data["bert_label"]) ??
        # Wait, bert_label is ONE dimensional per scalar? Or sequence?
        # Standard BERT MLM label: (B, Seq). 0 for ignored.
        
        # Original code:
        # predicted_index = torch.argmax(data["bert_label"]) <-- THIS IS WEIRD if label is indices.
        # If label is OneHot, simple argmax works. 
        # But usually labels are LongTensor of indices.
        # If labels are indices, argmax gives the position of maximum index?? No.
        
        # Let's assume standard PyTorch NLLLoss targets: LongTensor (B, Seq).
        # ignore_index=0.
        
        # Calculate accuracy on non-ignored tokens
        mask = (labels != 0) # Basic mask
        
        correct = (predictions == labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        self.log("val_acc", accuracy, prog_bar=True)
        return mask_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class BGCMultiLabelLightningModule(BaseLightningModule):
    def __init__(self, model, pos_weights, **kwargs):
        super().__init__(model, **kwargs)
        self.pos_weights = pos_weights
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

    def training_step(self, batch, batch_idx):
        model_output = self.model.forward(batch["bert_input"])
        loss = self.criterion(model_output, batch["classification_label"].float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.model.forward(batch["bert_input"])
        loss = self.criterion(model_output, batch["classification_label"].float())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        predictions = self.model.forward(batch["bert_input"])
        return torch.sigmoid(predictions)


class BGCRegressionLightningModule(BaseLightningModule):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.criterion = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        model_output = self.model.forward(batch["bert_input"])
        loss = self.criterion(model_output, batch["regression_value"].float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.model.forward(batch["bert_input"])
        loss = self.criterion(model_output, batch["regression_value"].float())
        self.log("val_loss", loss, prog_bar=True)
        return loss
        
    def predict_step(self, batch, batch_idx):
        return self.model.forward(batch["bert_input"])


class BGCMetricLightningModule(BaseLightningModule):
    def __init__(self, model, loss_type="correlation", **kwargs):
        super().__init__(model, **kwargs)
        self.loss_type = loss_type

    def correlation_loss(self, d_embed, d_target, eps=1e-8):
        # d_embed, d_target are shape (num_pairs,)
        d_embed_centered = d_embed - d_embed.mean()
        d_target_centered = d_target - d_target.mean()

        numerator = (d_embed_centered * d_target_centered).sum()
        denominator = torch.sqrt(
            (d_embed_centered**2).sum() * (d_target_centered**2).sum() + eps
        )
        corr = numerator / denominator
        return 1 - corr

    def cross_entropy_similarity_loss(self, z, target_sims, sigma=1.0, eps=1e-8):
        device = z.device
        B = z.size(0)
        # Pairwise squared distances
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        sqdist = (diff**2).sum(dim=-1)
        sim = torch.exp(-sqdist / (2.0 * sigma**2))
        
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(diag_mask, 0.0)
        
        row_sums_q = sim.sum(dim=1, keepdim=True) + eps
        q = sim / row_sums_q
        
        target_sims = target_sims.to(device).float()
        target_sims = target_sims.masked_fill(diag_mask, 0.0)
        row_sums_p = target_sims.sum(dim=1, keepdim=True) + eps
        p = target_sims / row_sums_p
        
        return -(p * (q + eps).log()).sum()

    def triplet_loss(self, anchor, pos, neg, margin=0.1):
        d_pos = ((anchor - pos) ** 2).sum(dim=-1).sqrt()
        d_neg = ((anchor - neg) ** 2).sum(dim=-1).sqrt()
        return torch.relu(d_pos - d_neg + margin).mean()

    def sample_triplets_indices(self, D, num_triplets, tau_pos=0.2, tau_neg=0.2):
        device = D.device
        N = D.shape[0]
        a = torch.randint(0, N, (num_triplets,), device=device)
        rows = D[a]
        
        # Positives
        tau_pos = max(float(tau_pos), 1e-6)
        pos_logits = -rows / tau_pos
        pos_logits[torch.arange(num_triplets, device=device), a] = -float("inf")
        pos_probs = torch.softmax(pos_logits, dim=1)
        p = torch.multinomial(pos_probs, 1).squeeze(1)
        
        # Negatives
        tau_neg = max(float(tau_neg), 1e-6)
        neg_logits = rows / tau_neg
        neg_logits[torch.arange(num_triplets, device=device), a] = -float("inf")
        neg_probs = torch.softmax(neg_logits, dim=1)
        n = torch.multinomial(neg_probs, 1).squeeze(1)
        
        return a.long(), p.long(), n.long()

    def loss_function(self, model_output, batch):
        if self.loss_type == "correlation":
            d_embed = pairwise_cosine_vector(model_output)
            d_target = pairwise_cosine_vector(batch["target_embedding"].float())
            return self.correlation_loss(d_embed, d_target)
        elif self.loss_type == "cross_entropy":
            sim = cosine_similarity_matrix(batch["target_embedding"].float())
            return self.cross_entropy_similarity_loss(model_output, sim)
        elif self.loss_type == "triplet":
            d_target = cosine_distance_matrix(batch["target_embedding"].float())
            a, p, n = self.sample_triplets_indices(d_target, 2048, tau_pos=0.1, tau_neg=0.1)
            return self.triplet_loss(model_output[a], model_output[p], model_output[n], margin=0.1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def training_step(self, batch, batch_idx):
        model_output = self.model.forward(batch["bert_input"])
        loss = self.loss_function(model_output, batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.model.forward(batch["bert_input"])
        loss = self.loss_function(model_output, batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss


class BGCTrainer:
    """
    Class for training a BGC Masked Language Model (MLM).
    Encapsulates data loading, model initialization, training, and saving.
    Refactored to use PyTorch Lightning.
    """

    def __init__(self, args_dict):
        self.args_dict = args_dict
        print("Arguments:")
        for k, v in self.args_dict.items():
            print(f"{k}: {v}")

        torch.manual_seed(self.args_dict["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(self.args_dict['thread_count'])

        self.unknown_threshold = self.args_dict["unknown_threshold"]
        self.max_bgc_length = self.args_dict["max_bgc_length"]
        self.training_datasets_file = self.args_dict["training_datasets_file"]
        self.model_name = self.args_dict["model_name"]
        
        self.bgc_tokens = None
        self.padded_bgcs = None
        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.bgc_mlm_model = None
        self.mlm = None
        self.token_list = ["PAD", "CLS", "SEP", "MASK", "UNK"]

    def init_model(self):
        if self.token_list is None:
            raise ValueError("Token list is not initialized. Run load_data() first.")

        self.bgc_mlm_model = BGC_MLM(
            vocab_size=len(self.token_list),
            seq_len=self.max_bgc_length,
            d_model=self.args_dict["d_model"],
            n_layers=self.args_dict["n_layers"],
            heads=self.args_dict["heads"],
            dropout=self.args_dict["dropout"]
        )

        self.mlm = MLM(self.bgc_mlm_model, len(self.token_list))
        # No need to move to device manually with Lightning, but can't hurt if passed to module

    def train(self):
        if self.train_set is None or self.mlm is None:
             raise ValueError("Data or Model not initialized.")

        train_loader = DataLoader(
            self.train_set, 
            batch_size=self.args_dict["batch_size"], 
            shuffle=True, 
            pin_memory=True,
            num_workers=4 
        )
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.args_dict["batch_size"],
            shuffle=False,
            num_workers=4
        )

        # Initialize Lightning Module
        pl_module = MLMLightningModule(
            self.mlm, 
            vocab_size=len(self.token_list),
            d_model=self.args_dict["d_model"],
            warmup_steps=1000 # Configurable?
        )

        # Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="auto",
            devices=1,
            enable_checkpointing=True,
            default_root_dir=os.getcwd()
        )

        print("Starting Training with PyTorch Lightning...")
        trainer.fit(pl_module, train_loader, val_loader)
        
        # Save Model State (Manual save as requested, though checkpoints exist)
        torch.save(self.mlm.state_dict(), self.model_name)
        print(f"Model saved to {self.model_name}")

    def save_output(self):
        # Lightning handles logging, but we can access trainer logs if needed
        # Or parse TensorBoard/CSV logs.
        pass