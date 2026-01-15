import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys
import os

from src.training.trainer import (
    BaseLightningModule,
    MLMLightningModule,
    BGCMultiLabelLightningModule,
    BGCRegressionLightningModule,
    BGCMetricLightningModule
)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.linear = nn.Linear(128, 128)
    def forward(self, x):
        return x

def test_instantiation():
    print("Testing BaseLightningModule...")
    model = DummyModel()
    base = BaseLightningModule(model, d_model=128)
    assert base.hparams.lr == 1e-4

    print("Testing MLMLightningModule...")
    # MLMLightningModule specific
    mlm_module = MLMLightningModule(model, vocab_size=100)
    
    print("Testing BGCMultiLabelLightningModule...")
    pos_weights = torch.ones(10)
    # Mocking usage requires model output shape matching loss, skipping forward test here
    multilabel = BGCMultiLabelLightningModule(model, pos_weights=pos_weights)

    print("Testing BGCRegressionLightningModule...")
    reg = BGCRegressionLightningModule(model)

    print("Testing BGCMetricLightningModule...")
    metric = BGCMetricLightningModule(model, loss_type="correlation")

    print("All modules instantiated successfully.")

if __name__ == "__main__":
    test_instantiation()
