import torch, torch.nn as nn

gpt_config = {
    'vocab_size': 50257,
    'context_length': 256, 
    'embed_dim': 1024, 
    'n_heads': 16, 
    'n_layers': 16, 
    'drop_rate': 0.1, 
    'qkv_bias': False
}

path = 'GPTs//GPT_1//save.pth'
def load(model, optimiser, path):
    checkpoint = torch.load(path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimiser_state'])


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



class test_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_embed = nn.Embedding(config['context_length'], config['embed_dim'])
        self.drop_embeds = nn.Dropout(config['drop_rate'])
        self.trf_blocks = nn.Sequential(*[transformer_block(config) for _ in range(config['n_layers'])])
        self.final_norm = test_layer_norm(config['embed_dim'])
        self.out_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias = False)

    def forward(self, in_idx):
        batch_size, sequence_len = in_idx.shape
        token_embeds = self.token_embed(in_idx)
        pos_embeds = self.pos_embed(torch.arange(sequence_len, device = in_idx.device))
        x = token_embeds + pos_embeds
        x = self.drop_embeds(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class transformer_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = multiHeadAttention(
            d_in = config['embed_dim'],
            d_out = config['embed_dim'],
            context_length = config['context_length'],
            num_heads = config['n_heads'],
            dropout = config['drop_rate'],
            qkv_bias = config['qkv_bias']
            )
        self.ff = feed_forward(config)
        self.norm1 = test_layer_norm(config['embed_dim'])
        self.norm2 = test_layer_norm(config['embed_dim'])
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    
class test_layer_norm(nn.Module):
    def __init__(self, embed_dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        variance = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(variance)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class feed_forward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['embed_dim'], 4 * config['embed_dim']), 
            GELU(), 
            nn.Linear(4 * config['embed_dim'], config['embed_dim']))

    def forward(self, x):
        return self.layers(x)
    


def text_to_token(text, tokeniser):
    encoded = tokeniser.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_to_text(tokens, tokeniser):
    flat = tokens.squeeze(0)
    return tokeniser.decode(flat.tolist())
    

#code dependencies  
import tiktoken
tokeniser = tiktoken.get_encoding('gpt2')
torch.manual_seed(123)

model = test_GPT(gpt_config)

optimiser = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
load(model, optimiser, path)

model.eval()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)



#GPT test 1
# batch = []
# text1 = 'Every effort moves you'
# text2 = 'Every day holds a'

# batch.append(torch.tensor(tokeniser.encode(text1)))
# batch.append(torch.tensor(tokeniser.encode(text2)))
# batch = torch.stack(batch, dim = 0)

# model = test_GPT(gpt_config)
# logits = model(batch)


#feed forward and activation test
# ffn = feeed_forward(gpt_config)
# x = torch.rand(2, 3, 768)
# out = ffn(x)
# print(out.shape)


#transformer test
# x = torch.rand(2, 4, 768)
# transformer = transformer_block(gpt_config)
# output = transformer(x)
# print(output)
# print(output.shape)


#GPT test 2
# model = test_GPT(gpt_config)
# out = model(batch)
# print('inputs:', batch)
# print(out.shape)
# print(out)
# total_params = sum(p.numel() for p in model.parameters())
# print('RAM usage: ', total_params / (1024 ** 2), ' MB')

#text test1
# from text_decode_test1 import generate_text
# start_context = 'Hello, I am'
# encoded = tokeniser.encode(start_context)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# out = generate_text(model = model, idx = encoded_tensor, max_new_tokens = 6, context_size = gpt_config['context_length'])
# decoded_text = tokeniser.decode(out.squeeze(0).tolist())
# print(decoded_text)

#text test 2
# from text_decode_test1 import generate_text
# tokens = generate_text(model = model, idx = text_to_token('Every effort moves you', tokeniser), max_new_tokens = 25, context_size = gpt_config['context_length'])
# print(token_to_text(tokens, tokeniser))

#text test 3
# from GPTs.GPT_2.text_gen_2 import generate
def generate_text(model, idx, max_new_tokens, context_size, temp = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temp > 0.0:
            logits = logits / temp
            probs = torch.softmax(logits, dim = -1)
           # print(torch.softmax(torch.flatten((torch.where(probs != 0))), dim = -1))
            idx_next = torch.multinomial(probs, num_samples = 1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)

        if idx == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim = -1)
        return idx
    

# tokens = generate(model = model, idx = text_to_token('Every effort moves you', tokeniser), 
#                   max_new_tokens = 25, context_size = gpt_config['context_length'], 
#                   top_k = 50, temp = 1.4)
# print(token_to_text(tokens, tokeniser))

# start = True
# text = 'hello world'
# while start:
#     response = input()
#     text += response
#     text += token_to_text(generate(model = model, idx = text_to_token(text, tokeniser), 
#                   max_new_tokens = 25, context_size = gpt_config['context_length'], 
#                   top_k = 50, temp = 1.4), tokeniser)
#     print(text)