import torch, torch.nn as nn

class multi_attention_wrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):         #effectively just initiates multiple causal attention heads and returns their outputs
        super().__init__()
        from causal_attention import causalAttention
        self.heads = nn.ModuleList([causalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim = -1)
    
from causal_attention import batch
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = multi_attention_wrapper(d_in, d_out, context_length, 0.0, num_heads = 2)
context_vectors = mha(batch)
print(context_vectors)                            #processed sequentially on line 10, improve by using matrix multiplication