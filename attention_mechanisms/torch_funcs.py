import torch
def dot(data, inputs):
    res = 0
    for idx, element in enumerate(inputs[0]):
        res += inputs[0][idx] * data[idx]
    return res

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)