import torch
import torch.nn as nn
from torch.nn import functional as F

config = {
    "vocab_size": 50304,          # All parameters must be a multiple of 64
    "context_length": 2048, 
    "embed_dim": 1024, 
    "hidden_dim": 2712,           # SwiGLU dim = dim * 8/3
    "n_layers": 24,               
    "n_heads": 16,                # dim / heads must = 64
    "n_kv_heads": 4,              # GQA ratio
    "norm_eps": 1e-5, 
    "rope_theta": 10000.0, 
    "drop_rate": 0.0, 
}

def load(model, optimiser, path):
    checkpoint = torch.load(path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimiser_state'])

class multiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), 'd_out must be divisible by num_heads'
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.c_attn = nn.Linear(d_in, 3 * d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias = False)
        
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

        freqs_cis = precompute_freqs_cis(self.head_dim, context_length)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x):
        B, T, C = x.size()

        # Calculate Q, K, V in a single pass, then split them
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_out, dim = 2)

        # Reshape for multi-head attention: (Batch, Heads, Sequence_Length, Head_Dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask = None, 
            dropout_p = self.dropout_p if self.training else 0.0, 
            is_causal = True
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))
    
class GQAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['n_heads']
        # Default to 1/4 of KV heads
        self.num_kv_heads = config.get('n_kv_heads', self.num_heads // 4) 
        self.head_dim = config['embed_dim'] // self.num_heads
        
        self.wq = nn.Linear(config['embed_dim'], self.num_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(config['embed_dim'], self.num_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(config['embed_dim'], self.num_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, config['embed_dim'], bias = False)
        
        self.resid_dropout = nn.Dropout(config['drop_rate'])

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        # Apply the complex rotation to Queries and Keys
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # Transpose for the attention backend: (Batch, Heads, Seq_len, Head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # GQA Step: Repeat the KV heads to match the number of Q heads
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k = torch.repeat_interleave(k, repeats = num_rep, dim = 1)
            v = torch.repeat_interleave(v, repeats = num_rep, dim = 1)
            
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask = None, 
            dropout_p = self.resid_dropout.p if self.training else 0.0, 
            is_causal = True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))

class feed_forward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['embed_dim'], 4 * config['embed_dim']), 
            nn.SwiGLU(), 
            nn.Linear(4 * config['embed_dim'], config['embed_dim']),
            nn.Dropout(config['drop_rate'])
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock_Modern(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config['embed_dim'])
        self.att = GQAAttention(config)
        self.norm2 = RMSNorm(config['embed_dim'])
        self.ff = SwiGLU(config)
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x, freqs_cis):
        # Pass freqs_cis down into the attention block
        x = x + self.drop_shortcut(self.att(self.norm1(x), freqs_cis))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x

class test_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embed = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.drop_embeds = nn.Dropout(config['drop_rate'])
        
        self.trf_blocks = nn.ModuleList([TransformerBlock_Modern(config) for _ in range(config['n_layers'])])
        self.final_norm = RMSNorm(config['embed_dim'])
        
        # Out head explicitly tied to token embeddings to save memory
        self.out_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias = False)
        self.out_head.weight = self.token_embed.weight

        # Precompute the RoPE frequencies as a persistent buffer so it isn't recalculated every pass
        freqs_cis = precompute_freqs_cis(
            dim = config['embed_dim'] // config['n_heads'], 
            seq_len = config['context_length']
        )
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, in_idx):
        # We don't slice the context length here, we slice the precomputed frequencies
        batch_size, sequence_len = in_idx.shape
        
        # Get embeddings
        x = self.drop_embeds(self.token_embed(in_idx))
        
        # Slice the frequencies to match the current sequence length
        freqs_cis = self.freqs_cis[:sequence_len]
        
        # Pass the sliced frequencies into every layer
        for block in self.trf_blocks:
            x = block(x, freqs_cis)
            
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        # No bias term (shift), only the scaling weight
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        # Calculate in float32 for stability, return in the input precision (bf16/fp16)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Scale by 8/3 to keep parameter parity with a standard FFN
        hidden_dim = int(config['embed_dim'] * 8 / 3)
        
        # Fused projection for the gating mechanism (W1 and W2)
        self.w12 = nn.Linear(config['embed_dim'], 2 * hidden_dim, bias = False)
        # Output projection (W3)
        self.w3 = nn.Linear(hidden_dim, config['embed_dim'], bias = False)
        self.drop = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        # Chunk splits the tensor in half along the last dimension
        x1, x2 = self.w12(x).chunk(2, dim = -1)
        # F.silu is PyTorch's native C++ implementation of Swish
        hidden = F.silu(x1) * x2
        return self.drop(self.w3(hidden))

def text_to_token(text, tokeniser):
    encoded = tokeniser.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_to_text(tokens, tokeniser):
    flat = tokens.squeeze(0)
    return tokeniser.decode(flat.tolist())

def generate_text(model, idx, max_new_tokens, context_size, temp = 0.0, top_k = None, eos_id = None):
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1, None]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temp > 0.0:
            logits = logits / temp
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim = -1)
        
    return idx

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # Calculate frequencies: theta^(-2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device = freqs.device, dtype = torch.float32)
    
    # Outer product: creating a (Sequence_Length, dim/2) matrix
    freqs = torch.outer(t, freqs)
    
    # Convert to complex numbers: polar to rectangular (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape Q and K into pairs of floats, then view as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast the frequencies to match batch and head dimensions: (1, Sequence, 1, Head_Dim/2)
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1]).to(xq_.device)
    
    # Complex multiplication rotates the vectors, then we flatten back to real numbers
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


import tiktoken
tokeniser = tiktoken.get_encoding('gpt2')
device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

model = test_GPT(config)
#load(model, optimiser, path)
model.eval()
model.to(device)


# tokens = generate_text(model = model, idx = text_to_token('Every effort moves you', tokeniser),
#                   max_new_tokens = 25, context_size = gpt_config['context_length'],
#                   top_k = 50, temp = 1.4)
# print(token_to_text(tokens, tokeniser))

# start = True
# context = 'hello world '
# while start:
#     response = input()
#     context += response + ' '
#     context += token_to_text(generate_text(model = model, idx = text_to_token(context, tokeniser),
#                   max_new_tokens = 25, context_size = gpt_config['context_length'],
#                   top_k = 50, temp = 1.4), tokeniser) + ' '
#     text = ' '.join(context.split()[-25:])
#     print(len(text.split()[-25:]))
#     print(text)