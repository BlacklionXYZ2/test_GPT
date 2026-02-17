import torch, torch.nn as nn

torch.manual_seed(123)
batch = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch)
mean = out.mean(dim = -1, keepdim = True)
var = out.var(dim = -1, keepdim = True)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim = -1, keepdim = True)
var = out_norm.var(dim = -1, keepdim = True)
print(out_norm)
print(mean)
print(var)