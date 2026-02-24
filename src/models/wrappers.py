import torch
from src.models.architecture import BGC_MLM, MaskedLanguageModel, MLM, BGCMultiLabelClassifier, BGCRegression, BGCMetricLearning
from src.models.trainers import ScheduledOptim, MLMTrainer, BGCMultiLabelTrainier, BGCRegressionTrainier, BGCMetricTrainer

def create_mlm_model(vocab_size, seq_len=20, d_model=768, n_layers=12, heads=12, dropout=0.1):
    """
    Creates and returns the combined MaskedLanguageModel (MLM object) using BGC_MLM inside.
    This replaces the two step instantiation logic seen previously.
    """
    bgc_mlm_model = BGC_MLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        heads=heads,
        dropout=dropout
    )
    return MLM(bgc_mlm_model, vocab_size)

# Export aliases to match BGC_MLM_tools exactly if they are called directly
__all__ = [
    "BGC_MLM",
    "MLM",
    "create_mlm_model",
    "BGCMultiLabelClassifier",
    "BGCRegression",
    "BGCMetricLearning",
    "ScheduledOptim",
    "MLMTrainer", 
    "BGCMultiLabelTrainier", 
    "BGCRegressionTrainier", 
    "BGCMetricTrainer"
]
