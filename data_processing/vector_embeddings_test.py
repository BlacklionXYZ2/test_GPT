import torch, dataloader as dl

vocab_size = 50257
output_dims = 256
embed_layer = torch.nn.Embedding(vocab_size, output_dims)

maxLength = 4
dataLoader = dl.createDataLoader(dl.rawText, batchSize = 8, maxLength = maxLength, stride = maxLength, shuffle = False)
data_iter = iter(dataLoader)
inputs, targets = next(data_iter)

token_embeds = embed_layer(inputs)

contextLength = maxLength
pos_embed_layer = torch.nn.Embedding(contextLength, output_dims)
pos_embeds = pos_embed_layer(torch.arange(contextLength))

input_embeds = token_embeds + pos_embeds
print(input_embeds) # embeds will be optimised in training