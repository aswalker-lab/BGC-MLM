# BGC-MLM
masked language foundation model for biosynthetic gene clusters
# Pretraining with a masked language task
Masked pretraining can be run with the following command:
<br/>
```python BGC_MLM_train.py <OUTPUT_NAME> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE>```
<br/>
The directory that you run BGC_MLM_train.py from must have a train.txt file that contains a list of paths to the tokenized BGCs (one per line), tar files containing tokenized BGCs are available at https://huggingface.co/datasets/allie-walker/BGC_MLM_data/tree/main, create train.txt so that paths match where ever you store this directory.
# Finetuning models
Multilabel Classification task:
<br/>
Classification can be trained with the following command:
```python classificationTask.py <PATH_TO_PRETRAINED_MODEL> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <FILE_WITH_BGC_LIST> <ClASSIFICATION_LABELS_FILE> <PATH_TO_TOKENIZED_BGCS>```
<br/>
By default this runs the training with weights for the language model encoder frozen. To run the training with unfrozen language model encoder weights add ```--freeze 0``` to the command. By default the model is output to <PATH_TO_PRETRAINED_MODEL>_regression. To give the model a different name, use the ```--model_output``` argument, this will add a custom suffix to the output filename. To specify the number of epochs add ```--epochs <NUMBER_OF_EPOCHS> to the command. By default this runs with class weighting, to not use class weighting add ```--use_pos_weights 0``` to the command.
<br/>
Regression task:
<br/>
Regression can be trained with the following command:
<br/>
```python regressionTask.py <PATH_TO_PRETRAINED_MODEL> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <FILE_WITH_BGC_LIST> <REGRESSION_LABELS_FILE> <PATH_TO_TOKENIZED_BGCS>```
<br/>
By default this runs the training with weights for the language model encoder frozen. To run the training with unfrozen language model encoder weights add ```--freeze 0``` to the command. By default the model is output to <PATH_TO_PRETRAINED_MODEL>_regression. To give the model a different name, use the ```--model_output``` argument, this will add a custom suffix to the output filename. To specify the number of epochs add ```--epochs <NUMBER_OF_EPOCHS> to the command.
# Training models from scratch
Classification task:
<br/>
```python classificationTaskFromScratch.py <OUTPUT_NAME> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <FILE_WITH_BGC_LIST> <CLASSIFICATION_LABELS_FILE> <PATH_TO_TOKENIZED_BGCS>```
<br/>
Regression task:
<br/>
Regression models can be trained from scratch with the following command:
<br/>
```python regressionTaksFromScratch.py <OUTPUT_NAME> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <FILE_WITH_BGC_LIST> <REGRESSION_LABELS_FILE> <PATH_TO_TOKENIZED_BGCS>```

# Calculating model metrics
Multilabel classification task:
<br/>
To calculate metrics for each label separately:
<br/>
```python classificationTaskTest.py <PATH_TO_TRAINED_MODEL> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <FILE_WITH_BGC_LIST> <CLASSIFICATION_LABELS_FILE> <PATH_TO_TOKENIZED_BGCS>```
<br/>
Or to calculate metrics across tasks (with both macro and micro averaging):
<br/>
```python classificationTaskTestOverallMetrics.py <PATH_TO_TRAINED_MODEL> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <FILE_WITH_BGC_LIST> <CLASSIFICATION_LABELS_FILE>
<PATH_TO_TOKENIZED_BGCS>```
<br/>

Regression metrics can be calculated for a test set using the command:
<br/>
```python regressionTaskTest.py <PATH_TO_TRAINED_MODEL> <UNKNOWN_THRESHOLD> <MAX_BGC_LENGTH> <D_MODEL> <N_LAYERS> <HEADS> <DROPOUT> <BATCH_SIZE> <FILE_WITH_BGC_LIST> <REGRESSION_LABELS_FILE> <PATH_TO_TOKENIZED_BGCS>```
