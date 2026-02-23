# Current Implementation Summary

## 1. Dataset Loading and Preprocessing

The system loads data by reading text files containing paths to individual sample files. Each sample file is processed line-by-line, splitting content by commas to generate tokens.

**Preprocessing Steps:**
*   **Tokenization:** Tokens are counted across the entire dataset. Tokens appearing fewer times than a configurable threshold are replaced with an "UNK" token.
*   **Padding:** Sequences are padded with "PAD" tokens to a fixed maximum length. This padding occurs in memory before dataset object creation.
*   **Data Structure:** All processed data (tokens, labels) is loaded entirely into memory as lists before being passed to dataset objects.

**Dataset Variations:**
The logic uses distinct dataset implementations for different tasks:
*   **Masked Language Modeling (MLM):** Applies dynamic masking to input sequences. 15% of tokens are selected for prediction. Of these, 80% are replaced with a [MASK] token, 10% are replaced with a random token, and 10% remain unchanged.
*   **Classification:** Stores tokenized sequences alongside multi-label classification targets.
*   **Regression:** Stores tokenized sequences alongside float regression values.
*   **Metric Learning:** Stores tokenized sequences alongside target embedding vectors.

## 2. Model Architectures

The core model is a Transformer Encoder architecture (BERT-style).

**Core Components:**
*   **Embeddings:** Combines learned token embeddings with fixed sinusoidal positional encodings.
*   **Encoder Blocks:** A stack of transformer encoder layers. Each layer consists of:
    *   Multi-Head Self-Attention
    *   Layer Normalization
    *   Position-wise Feed-Forward Network (Linear -> GELU -> Linear)
    *   Residual connections after attention and feed-forward blocks.

**Task-Specific Heads:**
*   **Masked Language Model:** A linear layer projecting the encoder output to the vocabulary size, followed by LogSoftmax.
*   **Classification:** Takes the mean of the encoder output across the sequence length, followed by a Linear layer, ReLU activation, and a final Linear projection to the number of classes.
*   **Regression:** Takes the mean of the encoder output, followed by a Linear layer, ReLU activation, and a final Linear projection to the regression targets.
*   **Metric Learning:** Takes the mean of the encoder output, followed by a Linear layer, ReLU activation, and a final Linear projection to the embedding dimension.

## 3. Model Training

Training logic is implemented separately for each task type, though they share a common structure.

**Training Loop:**
*   Iterates through epochs, processing data in batches.
*   Manages the forward pass, loss calculation, backpropagation, and optimizer steps explicitly within the loop.
*   Tracks training and validation loss, logging metrics at set intervals.

**Optimization:**
*   **Optimizer:** AdamW is used for optimization.
*   **Scheduling:** A custom learning rate scheduler (Noam scheme) adjusts the learning rate based on the model dimension and warmup steps.
    *   Linearly increases learning rate during warmup.
    *   Decays learning rate proportional to the inverse square root of step number thereafter.

**Loss Functions:**
*   **MLM:** Negative Log Likelihood Loss (NLLLoss), ignoring padding indices.
*   **Classification:** Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss), supporting positive class weighting.
*   **Regression:** Mean Squared Error Loss (MSELoss).
*   **Metric Learning:** Supports multiple loss types including Correlation Loss (1 - correlation coefficient), Cross Entropy on similarity matrices, and Triplet Loss.
