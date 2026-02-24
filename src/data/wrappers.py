import torch
import random
from torch.utils.data import Dataset
from src.data.datasets import BGCDataset

class MLMDataset(Dataset):
    def __init__(self, data, token_list, seq_len=20):
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
    
    def __len__(self):
        return self.corpus_lines
    
    def maskSequence(self, sequence, token_list):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
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
                    output.append(random.randrange(len(token_list)))
                else:
                    output.append(token_list.index(token))
                output_label.append(token_list.index(token))
            else:
                output.append(token_list.index(token))
                output_label.append(0)
        output = {"bert_input": output, "bert_label": output_label}
        return {key: torch.tensor(value) for key, value in output.items()}
    
    def get_sent(self, index):
        return self.lines[index]
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        bgc_object = self.maskSequence(bgc, self.token_list)
        return bgc_object

class specificMaskMLMDataset(Dataset):
    def __init__(self, data, token_list, mask_indices, seq_len=20):
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.mask_indices = mask_indices
    
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        return self.lines[index]
    
    def makeItem(self, sequence, token_list):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
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
        output = {"bert_input": output, "bert_label": output_label}
        return {key: torch.tensor(value) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        bgc_object = self.makeItem(bgc, self.token_list)
        return bgc_object

class UnmaskedBERTDataset(Dataset):
    def __init__(self, data, token_list, seq_len=20):
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
    
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        return self.lines[index]
    
    def makeItem(self, sequence, token_list):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue
            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {"bert_input": output, "bert_label": output_label}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        bgc_object = self.makeItem(bgc, self.token_list)
        return bgc_object

class BGCClassificationDatasets(Dataset):
    def __init__(self, data, token_list, classifications, seq_len=20):
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.classifications = classifications
        
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        return self.lines[index]
    
    def get_classification(self,index):
        return self.classifications[index]
    
    def makeItem(self, sequence, token_list, classification_label):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue
            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {"bert_input": output, "bert_label": output_label, "classification_label": classification_label}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        classification = self.get_classification(item)
        bgc_object = self.makeItem(bgc, self.token_list, classification)
        return bgc_object

class BGCRegressionDatasets(Dataset):
    def __init__(self, data, token_list, values, seq_len=20):
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.values = values
        
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        return self.lines[index]
    
    def get_values(self,index):
        return self.values[index]
    
    def makeItem(self, sequence, token_list, regression_value):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue
            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {"bert_input": output, "bert_label": output_label, "regression_value": regression_value}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        values = self.get_values(item)
        bgc_object = self.makeItem(bgc, self.token_list, values)
        return bgc_object

class BGCTargetEmbeddingDataset(Dataset):
    def __init__(self, data, token_list, target_embedding, seq_len=20):
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.target_embedding = target_embedding
        
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        return self.lines[index]
    
    def get_target_embedding(self,index):
        return self.target_embedding[index]
    
    def makeItem(self, sequence, token_list, target_embedding):
        output_label = []
        output = []
        for i, token in enumerate(sequence):
            if token == "SEP" or token == "CLS" or token == "PAD":
                output.append(token_list.index(token))
                output_label.append(token_list.index("PAD"))
                continue
            output.append(token_list.index(token))
            output_label.append(token_list.index(token))
        output = {"bert_input": output, "bert_label": output_label, "target_embedding": target_embedding}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        target_embedding = self.get_target_embedding(item)
        bgc_object = self.makeItem(bgc, self.token_list, target_embedding)
        return bgc_object
