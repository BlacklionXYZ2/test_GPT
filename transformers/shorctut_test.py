import torch, torch.nn as nn
from torch import tensor

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class deepNN(nn.Module):
    def __init__(self, layer_sizes, shortcut):
        super().__init__()
        self.use_shortcut = shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[x], layer_sizes[x + 1]), GELU()) 
            for x in range(0, len(layer_sizes) - 1)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            output = layer(x)
            if self.use_shortcut and x.shape == output.shape:
                x = x + output
            else:
                x = output
        return x
    

layer_sizes = [3, 3, 3, 3, 3, 1]
sample = tensor([[1., 0., 1.]])
torch.manual_seed(123)
model = deepNN(layer_sizes, shortcut = True)

def print_grad(model, x):
    output = model(x)
    target = tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} - grad - {param.grad.abs().mean().item()}')

print_grad(model, sample)