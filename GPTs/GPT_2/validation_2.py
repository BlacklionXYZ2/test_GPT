import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gpt_2 import gpt_config, model, tokeniser

file = 'input.txt'
try:
    with open(f'text//{file}', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print(f"Warning: text//{file} not found. Ensure the path is correct.")
    text = "Fallback text for testing. " * 1000

total_chars = len(text)
total_tokens = len(tokeniser.encode(text))

train_ratio = 0.9
split_idx = int(train_ratio * len(text))
training_data = text[:split_idx]
validation_data = text[split_idx:]

class MemoryEfficientGPTDataset(Dataset):
    def __init__(self, txt, tokeniser, maxLength, stride):
        # OPTIMIZATION 1: 1D Tensor Storage
        # Instead of appending thousands of overlapping chunked tensors into a Python list 
        # (which destroys System RAM), we store the entire text as ONE single 1D tensor 
        # and slice it dynamically on the fly.
        self.token_ids = torch.tensor(tokeniser.encode(txt), dtype=torch.long)
        self.maxLength = maxLength
        self.stride = stride

    def __len__(self):
        return (len(self.token_ids) - self.maxLength) // self.stride
    
    def __getitem__(self, idx):
        # Dynamically slice the 1D tensor when the DataLoader requests it
        start_idx = idx * self.stride
        inputChunk = self.token_ids[start_idx : start_idx + self.maxLength]
        targetChunk = self.token_ids[start_idx + 1 : start_idx + self.maxLength + 1]
        return inputChunk, targetChunk

def createDataLoader(txt, batchSize=4, maxLength=256, stride=128, shuffle=True, dropLast=True, numWorkers=0):
    tokeniser = tiktoken.get_encoding('gpt2')
    dataset = MemoryEfficientGPTDataset(txt, tokeniser, maxLength, stride)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batchSize, 
        shuffle=shuffle, 
        drop_last=dropLast, 
        num_workers=numWorkers,
        pin_memory=True # OPTIMIZATION 2: Pin Memory
        # This locks the data in a special area of your System RAM so it can be 
        # transferred across the PCIe bus to your GPU VRAM significantly faster.
    )
    return dataloader

# OPTIMIZATION 3: Cranking up the Batch Size and Workers
# Because we are saving so much VRAM now, we can feed 16 or 32 sequences at once.
# We also use numWorkers=2 so the CPU pre-fetches the next batch while the GPU does math.
training_loader = createDataLoader(
    training_data, 
    batchSize=16, 
    maxLength=gpt_config['context_length'], 
    stride=gpt_config['context_length'], 
    dropLast=True, 
    shuffle=True, 
    numWorkers=2 
)

validation_loader = createDataLoader(
    validation_data, 
    batchSize=16, 
    maxLength=gpt_config['context_length'], 
    stride=gpt_config['context_length'], 
    dropLast=True, 
    shuffle=False, # Validation data doesn't need to be shuffled
    numWorkers=2
)

def batch_calc_loss(input_batch, target_batch, model, device):
    # OPTIMIZATION 4: Non-Blocking Transfers
    # This allows the data transfer to overlap with GPU computation, saving precious milliseconds.
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0

    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for x, (input_batch, target_batch) in enumerate(data_loader):
        if x < num_batches:
            loss = batch_calc_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

# test code
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# with torch.no_grad():
#     train_loss = calc_loss_loader(training_loader, model, device)
#     valid_loss = calc_loss_loader(validation_loader, model, device)

# print(train_loss)
# print(valid_loss)