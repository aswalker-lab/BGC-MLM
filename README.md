# BGC-MLM
masked language foundation model for biosynthetic gene clusters
# Pretraining with a masked language task
Masked pretraining can be run with the following command:
<br/>
```python BGC_MLM_train.py <OUTPUT_NAME> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE>```
<br/>
The directory that you run BGC_MLM_train.py from must have a train.txt file that contains a list of paths to the tokenized BGCs (one per line), tar files containing tokenized BGCs are available at https://huggingface.co/datasets/allie-walker/BGC_MLM_data/tree/main, create train.txt so that paths match where ever you store this directory.
