import torch, torch.nn as nn

class causalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout = 0.5, qkv_bias = False):
        super().__init__()
        self.wquery = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wkey   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wvalue = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        queries = self.wquery(x)
        keys = self.wkey(x)
        values = self.wvalue(x)

        attnScores = queries @ keys.T

        context_length = attnScores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        
        masked = attnScores.masked_fill(mask.bool(), -torch.inf)

        attnWeights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim = -1) #to be continued with dropout implementation

        contextVector = attnWeights @ values
        return contextVector

from test_multiweight import inputs, dims_in, dims_out

torch.manual_seed(789)
test = causalAttention(dims_in, dims_out)
print(test(inputs))