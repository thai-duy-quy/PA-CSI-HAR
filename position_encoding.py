import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tensorflow.keras import layers

class PE(layers.Layer):
    def __init__(self, d_model, max_seq_length=500, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = layers.Dropout(dropout)
        
        pe = np.zeros((max_seq_length, d_model),dtype=float)

        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.reshape(1,pe.shape[0],pe.shape[1])      
        self.pe = pe
    
    def call(self, x):
        x = x*math.sqrt(self.d_model)
        seq_length = x.shape[1] # x.size(1)
        
        pe = self.pe[:, :seq_length] 
        x = x + pe
        x = self.dropout(x)
        return x

def normal_pdf(pos, mu, sigma):
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)

# Gaussian Position Endcoding (from THAT model) 
class GRE(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(GRE, self).__init__()
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(total_size)], requires_grad=False).unsqueeze(1).repeat(1, K)
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        temp = pos_enc.unsqueeze(0)
        return x + temp.detach().numpy()