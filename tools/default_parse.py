import argparse

def parse(script, args):
    """Parse arguments for a given script. Move default arguments to here."""
    try:
        return eval("_parse_" + script)(args)
    except:
        raise Exception("Invalid script name")
                
def _parse_BGC_train(args):
    """Parse arguments for BGC training."""
    parser = argparse.ArgumentParser()            
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

    args = parser.parse_args(args)
    return vars(args)