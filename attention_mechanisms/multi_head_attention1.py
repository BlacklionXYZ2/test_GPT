import torch, torch.nn as nn

class multiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            'd_out must be divisible by num_heads'
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.wq = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.wk(x)
        queries = self.wq(x)
        values = self.wv(x)

        keys    = keys.view(   b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view( b, num_tokens, self.num_heads, self.head_dim)

        keys    = keys.transpose(   1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose( 1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1, 2)

        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector


#test code 
# torch.manual_seed(123)

# inputs = torch.tensor([
#     [0.43, 0.15, 0.89], 
#     [0.55, 0.87, 0.66], 
#     [0.57, 0.85, 0.64], 
#     [0.22, 0.58, 0.33], 
#     [0.77, 0.25, 0.10], 
#     [0.05, 0.80, 0.55]
# ])
# batch = torch.stack((inputs, inputs), dim = 0)

# batch_size, context_len, d_in = batch.shape
# d_out = 24
# mha = multiHeadAttention(d_in, d_out, context_len, 0.0, num_heads = 12)
# context_vecs = mha(batch)
# print(context_vecs)