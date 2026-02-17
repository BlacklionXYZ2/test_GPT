import torch

inputText = 'Your journey starts with one step'
inputs = torch.tensor([
    [0.43, 0.15, 0.89], 
    [0.55, 0.87, 0.66], 
    [0.57, 0.85, 0.64], 
    [0.22, 0.58, 0.33], 
    [0.77, 0.25, 0.10], 
    [0.05, 0.80, 0.55]
])

i2 = inputs[1]
dims_in = inputs.shape[1]
dims_out = 2

torch.manual_seed(123)
w_q = torch.nn.Parameter(torch.rand(dims_in, dims_out), requires_grad = False)
w_k = torch.nn.Parameter(torch.rand(dims_in, dims_out), requires_grad = False)
w_v = torch.nn.Parameter(torch.rand(dims_in, dims_out), requires_grad = False)

q2 = i2 @ w_q
k2 = i2 @ w_k
v2 = i2 @ w_v

keys = inputs @ w_k
values = inputs @ w_v

attnScores = q2 @ keys.T

d_k = keys.shape[-1]
attnWeights = torch.softmax(attnScores / d_k ** 0.5, dim = -1)

contextVector = attnWeights @ values


import self_attentionV1 as sa1

torch.manual_seed(123)
sa_v1 = sa1.selfAttentionV1(dims_in, dims_out)
#print(sa_v1(inputs))

import self_attentionV2 as sa2

torch.manual_seed(789)
sa_v2 = sa2.selfAttention(dims_in, dims_out)
#print(sa_v2(inputs))