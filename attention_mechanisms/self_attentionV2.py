import torch, torch.nn as nn

class selfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.wquery = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wkey   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wvalue = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x):
        queries = self.wquery(x)       #matrix mult error, shapes not comaptible, no clue why because it works for parameter()
        keys = self.wkey(x)
        values = self.wvalue(x)

        attnScores = queries @ keys.T

        attnWeights = torch.softmax(attnScores / keys.shape[-1] ** 0.5, dim = -1)

        contextVector = attnWeights @ values
        return contextVector
