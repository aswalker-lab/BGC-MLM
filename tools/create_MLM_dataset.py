"""
To Do:
- remove redundancy by creating a superclass, keep compatibility by having the individual classes just call the generic class
- determine how best to control torch device setting
- improve the efficiency of the maskSequence function
- determine if this is implemented correctly, since traditional BERT pairs sentences and this model seems to just output a single sequence
- modify logic to add the special tokens on the fly, rather than performing that in the training 
    (ie. add CLS, SEP, and PAD; unsure if UNK should just be treated as a token or if it should recieve special treatment)
"""


import random
from torch import Dataset, tensor

class MLMDataset(Dataset):
    def __init__(self, data, token_list, seq_len=20):
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
    
    def __len__(self):
        return self.corpus_lines
    
    #selection 15% tokens for masking of those
    #80 replaced with MASK
    #10 random
    #10 unchanged
    def maskSequence(self, sequence, token_list):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            #don't change SEP or CLS
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue
            
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    output.append(token_list.index("MASK"))
                elif prob < 0.9:
                    #don't generate random SEP, CLS, or PADS?
                    output.append(random.randrange(len(token_list)))
                else:
                    output.append(token_list.index(token))
                output_label.append(token_list.index(token))
            else:
                output.append(token_list.index(token))
                output_label.append(0)
        output = {"bert_input": output,
              "bert_label": output_label}
        return {key: tensor(value) for key, value in output.items()}
    
    def get_sent(self, index):
        '''return random sentence'''
        return self.lines[index]
    
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        # Step 2: replace random words in sentence with mask / random words
        bgc_object = self.maskSequence(bgc, self.token_list)
        return bgc_object


class specificMaskMLMDataset(Dataset):
    '''version of the dataset with a specific position masked to test accuracy'''
    def __init__(self, data, token_list, mask_indices, seq_len=20):
        '''mask_indices indicates which index will be masekd, does not apply to special tokens'''
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.mask_indices = mask_indices
    
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        '''return random sentence'''
        return self.lines[index]
    
    def makeItem(self, sequence, token_list):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            #don't change SEP or CLS
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue
            if i in self.mask_indices:
                output.append(token_list.index("MASK"))
                output_label.append(token_list.index(token))
            else:
                output.append(token_list.index(token))
                output_label.append(0)
        output = {"bert_input": output,
              "bert_label": output_label}
        #return {key: torch.tensor(value).cuda() for key, value in output.items()}
        return {key: tensor(value) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        bgc_object = self.makeItem(bgc, self.token_list)
        # Step 2: replace random words in sentence with mask / random words
        return bgc_object

class UnmaskedBERTDataset(Dataset):
    '''Version of dataset with no mask'''
    def __init__(self, data, token_list, seq_len=20):
        #self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
    
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        '''return random sentence pair'''
        return self.lines[index]
    
    def makeItem(self, sequence, token_list):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            #don't change SEP or CLS
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue
             
            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {"bert_input": output,
              "bert_label": output_label}
        #return {key: torch.tensor(value).cuda() for key, value in output.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: tensor(value).to(device) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        bgc_object = self.makeItem(bgc, self.token_list)
        # Step 2: replace random words in sentence with mask / random words
        return bgc_object