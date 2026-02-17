import tiktoken, torch, torch.nn as nn

tokeniser = tiktoken.get_encoding('gpt2')

file = 'harry_potter.txt'
with open(f'text//{file}', 'r', encoding = 'utf-8') as f:
    text = f.read()

total_chars = len(text)
total_tokens = len(tokeniser.encode(text))

train_ratio = 0.9
split_idx = int(train_ratio * len(text))
training_data = text[:split_idx]
validation_data = text[split_idx:]


from gpt_1 import gpt_config, model
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, txt, tokeniser, maxLength, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokeniser.encode(txt)

        for x in range(0, len(token_ids) - maxLength, stride):
            inputChunk = token_ids[x: x + maxLength]
            targetChunk = token_ids[x + 1: x + maxLength + 1]
            self.input_ids.append(torch.tensor(inputChunk))
            self.target_ids.append(torch.tensor(targetChunk))


    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    


def createDataLoader(txt, batchSize = 4, maxLength = 256, stride = 128, shuffle = True, dropLast = True, numWorkers = 0):
    tokeniser = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(txt, tokeniser, maxLength, stride)
    dataloader = DataLoader(dataset, batch_size = batchSize, shuffle = shuffle, drop_last = dropLast, num_workers = numWorkers)
    return dataloader


training_loader = createDataLoader(
    training_data, 
    batchSize = 2, 
    maxLength = gpt_config['context_length'], 
    stride = gpt_config['context_length'], 
    dropLast = True, 
    shuffle = True, 
    numWorkers = 0 
)

validation_loader = createDataLoader(
    validation_data, 
    batchSize = 2, 
    maxLength = gpt_config['context_length'], 
    stride = gpt_config['context_length'], 
    dropLast = True, 
    shuffle = True, 
    numWorkers = 0
)

def batch_calc_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
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



#test code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# with torch.no_grad():
#     train_loss = calc_loss_loader(training_loader, model, device)
#     valid_loss = calc_loss_loader(validation_loader, model, device)

# print(train_loss)
# print(valid_loss)