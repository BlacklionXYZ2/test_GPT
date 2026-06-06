import tiktoken, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import Dataset, DataLoader
from gpt_2 import gpt_config, model, tokeniser

file_path = 'train_data.bin'

try:
    full_data = np.memmap(file_path, dtype=np.uint16, mode='r')
except FileNotFoundError:
    print(f"Error: {file_path} not found. Have you run the tokenizer script?")
    exit()

val_tokens = 5_000_000 
split_idx = len(full_data) - val_tokens

training_data = full_data[:split_idx]
validation_data = full_data[split_idx:]

class MemmapGPTDataset(Dataset):
    def __init__(self, data_view, maxLength, stride):
        self.data_view = data_view
        self.maxLength = maxLength
        self.stride = stride

    def __len__(self):
        return (len(self.data_view) - self.maxLength) // self.stride
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        
        chunk = self.data_view[start_idx : start_idx + self.maxLength + 1]
        
        inputChunk = torch.from_numpy(chunk[:-1].astype(np.int64))
        targetChunk = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return inputChunk, targetChunk

def createDataLoader(data_view, batchSize=4, maxLength=256, stride=128, shuffle=True, dropLast=True, numWorkers=0):
    dataset = MemmapGPTDataset(data_view, maxLength, stride)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batchSize, 
        shuffle=shuffle, 
        drop_last=dropLast, 
        num_workers=numWorkers,
        pin_memory=True
    )
    return dataloader

training_loader = createDataLoader(
    training_data, 
    batchSize=16, 
    maxLength=gpt_config['context_length'], 
    stride=gpt_config['context_length'], 
    dropLast=True, 
    shuffle=True, 
    numWorkers=4  # Increased to 4 to handle background disk reads
)

validation_loader = createDataLoader(
    validation_data, 
    batchSize=16, 
    maxLength=gpt_config['context_length'], 
    stride=gpt_config['context_length'], 
    dropLast=True, 
    shuffle=False, 
    numWorkers=4
)

def batch_calc_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    
    logits = model(input_batch)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
    
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