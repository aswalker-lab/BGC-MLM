# BGC-MLM
masked language foundation model for biosynthetic gene clusters
# Pretraining with a masked language task
Masked pretraining can be run with the following command:
<br/>
```python BGC_MLM_train.py <OUTPUT_NAME> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE>```
<br/>
The directory that you run BGC_MLM_train.py from must have a train.txt file that contains a list of paths to the tokenized BGCs (one per line), tar files containing tokenized BGCs are available at https://huggingface.co/datasets/allie-walker/BGC_MLM_data/tree/main, create train.txt so that paths match where ever you store this directory.
# Finetuning Models
Classification task:
<br/>
Regression task:
<br/>
Regression can be trained with the following command:
<br/>
```python regressionTask.py <PATH_TO_PRETRAINED_MODEL> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <REGRESSION_LABELS_FILE> <PATH_TO_TOKENIZED_BGCS>```
<br/>
By default this runs the training with weights for the language model encoder frozen. To run the training with unfrozen language model encoder weights add ```--freeze 0``` to the command. By default the model is output to <PATH_TO_PRETRAINED_MODEL>_regression. To give the model a different name, use the ```--model_output`` argument, this will add a custom suffix to the output filename.
# Model metrics
