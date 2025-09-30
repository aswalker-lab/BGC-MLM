# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 16:54:14 2025

@author: Allison Walker
"""
#adapted from https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.compose import ColumnTransformer
import random
from pathlib import Path
import torch
import math
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
from torch.utils.data import Dataset, DataLoader

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
        return {key: torch.tensor(value) for key, value in output.items()}
    
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
        return {key: torch.tensor(value) for key, value in output.items()}
    
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
        return {key: torch.tensor(value).to(device) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        bgc_object = self.makeItem(bgc, self.token_list)
        # Step 2: replace random words in sentence with mask / random words
        return bgc_object
        
class PositionalEmbedding(torch.nn.Module):
     def __init__(self, d_model, max_len=128):
        super().__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        # include the batch size
        self.pe = pe.unsqueeze(0)   
        # self.register_buffer('pe', pe)
     def forward(self, x):
        return self.pe
    
class BGCClassificationDatasets(Dataset):
    '''Dataset for classification task'''
    def __init__(self, data, token_list, classifications, seq_len=20):
        #self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_list = token_list
        self.corpus_lines = len(data)
        self.lines = data
        self.classifications = classifications
    def __len__(self):
        return self.corpus_lines
    
    def get_sent(self, index):
        '''return random sentence pair'''
        return self.lines[index]
    
    def get_classification(self,index):
        return self.classifications[index]
    
    def makeItem(self, sequence, token_list,classification_label):
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
              "bert_label": output_label,
              "classification_label":classification_label}
        #return {key: torch.tensor(value).cuda() for key, value in output.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {key: torch.tensor(value).to(device) for key, value in output.items()}
    
    def __getitem__(self, item):
        bgc = self.get_sent(item)
        classification = self.get_classification(item)
        bgc_object = self.makeItem(bgc, self.token_list,classification)
        # Step 2: replace random words in sentence with mask / random words
        return bgc_object

class MLMEmbedding(torch.nn.Module):
    """
    MLM Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of MLMEmbedding
    """
    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, sequence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self.token(sequence).to(device) + self.position(sequence).to(device) 
        return self.dropout(x)
    
class MultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)  
        
        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)
    
class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class EncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        d_model=768,
        heads=12, 
        feed_forward_hidden=768 * 4, 
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    
class BGC_MLM(torch.nn.Module):
    """
    BGC-MLM model : Masked langauage model for BGCs
    """
    def __init__(self, vocab_size, seq_len=20, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = MLMEmbedding(vocab_size=vocab_size, seq_len=seq_len, embed_size=d_model)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])
    def forward(self, x):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x

class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class MLM(torch.nn.Module):
    """
    Masked Language Model
    """

    def __init__(self, bgc_mlm: BGC_MLM, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bgc_mlm = bgc_mlm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.mask_lm = MaskedLanguageModel(self.bgc_mlm.d_model, vocab_size)
        self.mask_lm = torch.nn.DataParallel(self.mask_lm)
        self.mask_lm.to(device)


    def forward(self, x):
        x = self.bgc_mlm(x)
        return self.mask_lm(x)    

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        if self.n_current_steps < self.n_warmup_steps:
            return np.power(self.n_warmup_steps, -1.7) * self.n_current_steps
        else:
            return np.min([np.power(self.n_current_steps, -0.5), np.power(self.n_warmup_steps,-0.7)])
        #return np.min([
         #   np.power(self.n_current_steps, -0.2),
          #  np.power(self.n_warmup_steps, -1.2) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        print("current steps: " + str(self.n_current_steps))
        print("lr: " + str(lr))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class MLMTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        val_data,
        test_dataloader=None, 
        lr= 1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        log_freq=10,
        device='cpu'
        ):

        self.device = device
        print(device)
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_data
        
        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bgc_mlm.d_model, n_warmup_steps=warmup_steps
            )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        self.train_loss_list = []
        self.val_loss_list= []
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data, self.val_data)

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.train_data, self.val_data, train=False)
    
    def validate(self, validation_set, batch_size=2144):
        val_loader = DataLoader(validation_set, batch_size)
        print('validating...')
        data_iter = tqdm.tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            bar_format="{l_bar}{r_bar}"
        )
        avg_loss = 0.0
        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                #print(data)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                #print(data)
                mask_lm_output = self.model.forward(data["bert_input"])
                
                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                
                avg_loss += mask_loss.item()
        
        return avg_loss / (i + 1)
    
    def predictSequence(self):
        data_iter = tqdm.tqdm(
            enumerate(self.train_data),
            total=len(self.train_data),
            bar_format="{l_bar}{r_bar}"
        )
        prediction_list = []
        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_10_accuracy = 0.0
        avg_50_accuracy = 0.0
        self.model.eval()
        total_data_points = 0
        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                #print(data)
                data = {key: value.to(self.device) for key, value in data.items()}
               
                predicted_index = torch.argmax(data["bert_label"])
                #this should occur if the sequence is PADed?
                if data["bert_label"][0][predicted_index] == 0:
                    #print(data["bert_label"][0])
                    #print("HERE!!")
                    continue
                # 1. forward the next_sentence_prediction and masked_lm model
                #print(data)
                mask_lm_output = self.model.forward(data["bert_input"])
                #print(mask_lm_output)
                
                # 2-1. NLL(negative log likelihood) loss of is_next classification result
                #next_loss = self.criterion(next_sent_output, data["is_next"])

                # 2-2. NLLLoss of predicting masked token word
                # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
                # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
                mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
                #print(mask_lm_output.transpose(1, 2))
                #print(data["bert_label"])
                print("mask loss " + str(mask_loss))

                # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                #loss = next_loss + mask_loss
                loss = mask_loss
                avg_loss += mask_loss.item()
                prediction_list.append(mask_lm_output)
                max_indices = torch.argmax(mask_lm_output,dim=2)
                
                total_data_points += 1
                if max_indices[0][predicted_index] == data["bert_label"][0][predicted_index]:
                    avg_accuracy += 1
                #rank predictions at position for top 10 and top 50 metric
                top10_val, top10_ind = torch.topk(mask_lm_output, k=10, dim=2)
                top50_val, top50_ind = torch.topk(mask_lm_output, k=50, dim=2)
                
                if max_indices[0][predicted_index] in top10_ind[:,predicted_index,:]:
                    avg_10_accuracy += 1
                if max_indices[0][predicted_index] in top50_ind[:,predicted_index,:]:
                    avg_50_accuracy += 1
                #print("max indices")
                #print(max_indices)
        if total_data_points > 0:
            avg_loss = avg_loss / total_data_points
            avg_accuracy = avg_accuracy/total_data_points
            avg_10_accuracy = avg_10_accuracy/total_data_points
            avg_50_accuracy = avg_50_accuracy/total_data_points
            print("average loss: " + str(avg_loss))
            print("avg top 10: " + str(avg_10_accuracy))
            return avg_loss, prediction_list, avg_accuracy, avg_10_accuracy, avg_50_accuracy
        else:
            return None, prediction_list, None, None, None
    
    def encode(self, data_loader):
        # progress bar
        self.model.eval()
        encoded_list = []
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            #print(data)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            #print(data)
            mask_lm_output = self.model.forward(data["bert_input"])
            encoder_output1 = self.model.bert.forward(data["bert_input"])
            #print("HERE!")
            #print(encoder_output1.shape)
            #print(encoder_output1)
            np_output = encoder_output1.detach().cpu().numpy()
            np_output = np.copy(np_output)
            #print(np_output.shape)
            pos_average = np.average(np_output,axis=1)
            #print(pos_average.shape)
            encoded_list.append(pos_average)
            #np_output = np_output.flatten()
            #print("NP SIZE!")
            #print(np_output.size)
            #print(np_output[0].size)
            #print(np_output[0])
            
        return encoded_list
        
    def iteration(self, epoch, data_loader, val_data, train=True):
        
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            #print(data)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            #print(data)
            mask_lm_output = self.model.forward(data["bert_input"])
            #print(mask_lm_output)
            
            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            #next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            #loss = next_loss + mask_loss
            loss = mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            #correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            #total_correct += correct
            #total_element += data["is_next"].nelement()
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)},\
                        val_loss={val_loss}"
                    )    
        else:
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)}"
                    )    
    def iterationPredict(self, epoch, data_loader, val_data, train=False):
        
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            #print(data)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            #print(data)
            mask_lm_output = self.model.forward(data["bert_input"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            #next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            #loss = next_loss + mask_loss
            loss = mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            #correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            #total_correct += correct
            #total_element += data["is_next"].nelement()
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train:
            val_loss = self.validate(val_data, batch_size=64)
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(avg_loss / len(data_iter))
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)},\
                        val_loss={val_loss}"
                    )    
        else:
            print(
                f"EP{epoch}, {mode}: \
                    avg_loss={avg_loss / len(data_iter)}"
                    )   
