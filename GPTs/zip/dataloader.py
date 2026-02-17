import tiktoken, torch
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


#test code
# with open('EPQ_projects//training.txt', 'r', encoding = 'utf-8') as f:
#     rawText = f.read()

# dataloader = createDataLoader(rawText, batchSize = 8, maxLength = 4, stride = 4, shuffle = False)
# data_iter = iter(dataloader)
# batch1 = next(data_iter)
# batch2 = next(data_iter)
# print(batch1)
# print(batch2)