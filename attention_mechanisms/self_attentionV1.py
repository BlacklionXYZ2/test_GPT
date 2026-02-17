import torch
import torch.nn as nn

class selfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.wquery = nn.Parameter(torch.rand(d_in, d_out))
        self.wkey = nn.Parameter(torch.rand(d_in, d_out))
        self.wvalue = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.wkey
        queries = x @ self.wquery
        values = x @ self.wvalue

        attnScores = queries @ keys.T

        attnWeights = torch.softmax(attnScores / keys.shape[-1] ** 0.5, dim = -1)

        contextVector = attnWeights @ values
        return contextVector
    