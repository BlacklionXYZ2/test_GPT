import torch, torch_funcs as tf

inputText = 'Your journey starts with one step'
inputs = torch.tensor([
    [0.43, 0.15, 0.89], 
    [0.55, 0.87, 0.66], 
    [0.57, 0.85, 0.64], 
    [0.22, 0.58, 0.33], 
    [0.77, 0.25, 0.10], 
    [0.05, 0.80, 0.55]
])

query = inputs[1]
attnScores = torch.empty(inputs.shape[0])
for i, xi in enumerate(inputs):
    attnScores[i] = torch.dot(xi, query)

attnWeights_ = tf.softmax(attnScores)
attnWeights = torch.softmax(attnScores, dim = 0)

contextVector = torch.zeros(query.shape)
for i, xi in enumerate(inputs):
    contextVector += attnWeights[i] * xi
print(contextVector)