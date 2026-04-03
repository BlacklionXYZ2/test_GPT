import torch
import torch.nn as nn
from torch.nn import functional as F

gpt_config = {
    'vocab_size': 50257,
    'context_length': 256, 
    'embed_dim': 1024, 
    'n_heads': 16, 
    'n_layers': 16, 
    'drop_rate': 0.1, 
    'qkv_bias': False
}

def load(model, optimiser, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimiser_state'])

class multiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), 'd_out must be divisible by num_heads'
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # OPTIMIZATION 1: Fused QKV projection. 
        # One large matrix multiplication is much faster on a GPU than three smaller ones.
        self.c_attn = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()

        # Calculate Q, K, V in a single pass, then split them
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_out, dim=2)

        # Reshape for multi-head attention: (Batch, Heads, Sequence_Length, Head_Dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # OPTIMIZATION 2: FlashAttention via PyTorch Native SDPA
        # This automatically dispatches to ROCm's optimized backends. 
        # `is_causal=True` applies the triangle mask natively without consuming NxN memory.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout_p if self.training else 0.0, 
            is_causal=True
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.out_proj(y))

class feed_forward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['embed_dim'], 4 * config['embed_dim']), 
            nn.GELU(), # OPTIMIZATION 3: PyTorch native GELU (Fused, no intermediate memory bloat)
            nn.Linear(4 * config['embed_dim'], config['embed_dim']),
            nn.Dropout(config['drop_rate'])
        )

    def forward(self, x):
        return self.layers(x)

class transformer_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # OPTIMIZATION 4: PyTorch native LayerNorm (Highly optimized C++ kernel)
        self.norm1 = nn.LayerNorm(config['embed_dim'])
        self.att = multiHeadAttention(
            d_in=config['embed_dim'],
            d_out=config['embed_dim'],
            context_length=config['context_length'],
            num_heads=config['n_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias']
        )
        self.norm2 = nn.LayerNorm(config['embed_dim'])
        self.ff = feed_forward(config)
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        # Pre-norm architecture remains unchanged (This is good!)
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x

class test_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_embed = nn.Embedding(config['context_length'], config['embed_dim'])
        self.drop_embeds = nn.Dropout(config['drop_rate'])
        self.trf_blocks = nn.Sequential(*[transformer_block(config) for _ in range(config['n_layers'])])
        self.final_norm = nn.LayerNorm(config['embed_dim'])
        self.out_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, sequence_len = in_idx.shape
        
        # Safety check
        assert sequence_len <= self.config['context_length'], "Sequence length exceeds context length"
        
        token_embeds = self.token_embed(in_idx)
        pos_embeds = self.pos_embed(torch.arange(sequence_len, device=in_idx.device))
        
        x = self.drop_embeds(token_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def text_to_token(text, tokeniser):
    encoded = tokeniser.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_to_text(tokens, tokeniser):
    flat = tokens.squeeze(0)
    return tokeniser.decode(flat.tolist())

def generate_text(model, idx, max_new_tokens, context_size, temp=0.0, top_k=None, eos_id=None):
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
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=-1)
        
    return idx

#dependencies
import tiktoken
tokeniser = tiktoken.get_encoding('gpt2')

model = test_GPT(gpt_config)
#load(model, optimiser, path)

model.eval()
device = ('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)