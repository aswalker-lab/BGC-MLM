import argparse


def parse(script, args) -> argparse.Namespace:
    """Parse arguments for a given script. Move default arguments to here."""
    parser = argparse.ArgumentParser()    
    script_map = {
        "BGC_train": lambda parser, args: _parse_BGC_train(parser, args),
        "classificationTask": lambda parser, args: _classificationTask(parser, args)
    }
    try:
        return script_map[script](parser, args)
    except:
        raise Exception("Invalid script name")
            
def _parse_BGC_train(parser, args) -> argparse.Namespace:
    """Parse arguments for BGC training."""
            
    parser.add_argument('model_name', required=True, help="Name of the model to be saved")                           
    parser.add_argument('unknown_threshold',type=int, required=True, help="Number of times a token must appear to be included in the vocabulary")
    parser.add_argument('max_bgc_length',type=int, required=True, help="Maximum length of BGCs to be included in the training data") 
    parser.add_argument('d_model',type=int, required=True, help="Embedding size for token embedding")                    
    parser.add_argument('n_layers',type=int, required=True, help="Number of layers in BERT")
    parser.add_argument('heads',type=int, required=True, help="Number of attention heads")                        
    parser.add_argument('dropout',type=float, required=True, help="Dropout rate (percent)")                    
    parser.add_argument('batch_size',type=int, required=True, help="Batch size")                   
    parser.add_argument('training_datasets_file', type=str, required=True, help="File containing list of training datasets")      
        
    parser.add_argument('--seed', type=int,default=0, required=False, help="Random seed")           
    parser.add_argument('--thread_count', type=int, default=8, required=False, help="Number of threads")  

    return parser.parse_args(args)

def _classificationTask(parser, args) -> argparse.Namespace:
    parser.add_argument('model_name')           # positional argument
    parser.add_argument('unknown_threshold',type=int)           # positional argument
    parser.add_argument('max_bgc_length',type=int)           # positional argument
    parser.add_argument('d_model',type=int)           # positional argument
    parser.add_argument('n_layers',type=int)           # positional argument
    parser.add_argument('heads',type=int)           # positional argument
    parser.add_argument('droupout',type=float)           # positional argument
    parser.add_argument('batch_size',type=int) #batch size
    parser.add_argument('data_set',type=str) #path to dataset file with features
    parser.add_argument('classification_file',type=str) # file containing classifications for BGCs
    parser.add_argument('token_path',type=str) #path to directory with tokens (not used in antiSMASH)
    parser.add_argument('--seed',type=int,default=0) #random seed
    parser.add_argument('--model_output',type=str,default="classification") #output name for classifier (not used by test, metrics, or antismash)
    parser.add_argument('--freeze',type=int,default=1) #freeze pretrained weights? (not used by test, metrics, or antismash)
    parser.add_argument('--epochs',type=int,default=50) #output name for classifier (not used by test, metrics, or antismash)
    parser.add_argument('--use_pos_weights',type=int,default=1) #if 0, set all pos_weights to 1 (not used by test, metrics, or antismash)
    parser.add_argument('--write_metrics',type=int,default=1) #if 0, do not write metrics after training (not used by test, metrics, or antismash)
    parser.add_argument('--train_fraction',type=float,default=0.9) #fraction to use for training, remaining will be val (not used by test, metrics, or antismash)
    
    return parser.parse_args(args)
