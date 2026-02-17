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

attnScores = torch.empty(6, 6)                     #attention scores are created by taking the dot product of two given input vectors
for i, xi in enumerate(inputs):                    #this creates effectively a table that shows the relationship between any two words
    for j, xj in enumerate(inputs):                #where greater values show a stronger relationship or similarity
        attnScores[i, j] = torch.dot(xi, xj)       #the softmax() function is then used to normalise these probabilities so they sum to 1
                                                   #those probabilities are then multiplied as a matrix against the input matrix to give the final context vectors. 
# or                                               #the input vectors were represented in 3 dimensions so the context vectors output in 3 dims #written 10.11.25

attnScores2 = torch.empty(6, 6)
attnScores2 = inputs @ inputs.T   # matrix multiplication


attnWeights = torch.softmax(attnScores2, dim = -1)

contextVecs = attnWeights @ inputs
print(contextVecs)