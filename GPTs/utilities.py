import torch, torch.nn as nn

torch.manual_seed(123)

def text_to_token(text, tokeniser):
    encoded = tokeniser.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_to_text(tokens, tokeniser):
    flat = tokens.squeeze(0)
    return tokeniser.decode(flat.tolist())

#test code
import  tiktoken 
from GPT_1 import gpt_1
from GPT_1.text_decode_test1 import generate_text
context = 'Every effort moves you'
tokeniser = tiktoken.get_encoding('gpt2')

token_ids = generate_text(model = gpt_1.model, 
                          idx = text_to_token(context, tokeniser), 
                          max_new_tokens = 10, 
                          context_size = gpt_1.gpt_config['context_length'])


inputs = torch.tensor([[16833, 3626, 6100], 
                       [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345], 
                        [1107, 588, 11311]])

with torch.no_grad():
    logits = gpt_1.model(inputs)
probs = torch.softmax(logits, dim = -1)
token_ids = torch.argmax(probs, dim = -1, keepdim = True)

text_idx = 0
target_probs_1 = probs[text_idx, [0, 1, 2], targets[text_idx]]
text_idx = 1
target_probs_2 = probs[text_idx, [0, 1, 2], targets[text_idx]]

log_probs = torch.log(torch.cat((target_probs_1, target_probs_2)))
avg_log_prob = torch.mean(log_probs)
neg_avg_log_prob = avg_log_prob  -1                          #loss through averages


logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

loss = nn.functional.cross_entropy(logits_flat, targets_flat) #loss through torch (better)


perplexity = torch.exp(loss)                                  #a measure of the number of tokens that the model may produce with this input (lower is better, to a point)




next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probs = torch.softmax(next_token_logits, dim = 0)
next_token_id = torch.multinomial(probs, num_samples = 1).item()           # better than argmax() because it introduces variance into the token selection

#temperature scaling
def temp_scale(logits, temp):
    scaled_logits = logits / temp
    return torch.softmax(scaled_logits, dim = 0) # this affects how flat the probability graph is, as temp increases, the sharpness decreases

#top-k sampling                -                temperature scaling can cause issues since it also increases the probabilities of contextually irrelevant tokens appearing, so we can restrict the domain to just the most probable tokens
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
new_logits = torch.where(condition = next_token_logits < top_logits[-1], input = torch.tensor(float('-inf')), other = next_token_logits)
top_k_probs = torch.softmax(new_logits, dim = 0)
