import torch, torch.nn as nn

class causalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):       #causal attention model finished 21/11/25
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.wquery = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wkey   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wvalue = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.wkey(x)
        queries = self.wquery(x)
        values = self.wvalue(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attnWeights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        attnWeights = self.dropout(attnWeights)

        context_vector = attnWeights @ values
        return context_vector
    
from test_multiweight import inputs, dims_in, dims_out
batch = torch.stack((inputs, inputs), dim = 0)
torch.manual_seed(123)
context_length = batch.shape[1]
ca = causalAttention(dims_in, dims_out, context_length, 0.0)
context_vecs = ca(batch)
print(context_vecs.shape)
print(context_vecs)