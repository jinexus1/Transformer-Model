import numpy as np
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    # Dropout is used to prevent overfitting
    def __init__(self, d_model:int,seq_len:int, dropout=float)->None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.d_model=d_model
        self.seq_len=seq_len
        # Create a matrix of shape (seq_len, d_model)
        # positional Encoding
        pe=torch.zeros(seq_len, d_model)
        #Create a vector of shape
        position=torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        #Applying sin for even positions
        pe[:,0::2]=torch.sin(position*div_term)
        #Applying cosin for odd positions
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)#make a tensor of size (1,seq_len,d_model)
        #to save tensor with the file
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps 
        self.alpha=nn.Parameter(torch.ones(1))#Multiplied
        self.bias=nn.Parameter(torch.zeros(1))#Added
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias

class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff)->None:
        super().__init__()
        self.linear_1=nn.Linear(d_model, d_ff)#W1  and B1
        self.dropout=nn.Dropout(dropout)


