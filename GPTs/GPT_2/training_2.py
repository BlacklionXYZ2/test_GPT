import torch, math, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from validation_2 import calc_loss_loader, training_loader, validation_loader, batch_calc_loss
from gpt_2 import text_to_token, token_to_text, generate_text, model, device, tokeniser

path = 'GPTs//GPT_2//save.pth'

def save(model, optimiser, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict()
        }, path)
    
def load(model, optimiser, path):
    checkpoint = torch.load(path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimiser_state'])

def save_checkpoint(model, optimiser, scaler, step, tokens_seen, path):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict(),
        'scaler_state': scaler.state_dict(), 
        'step': step,
        'tokens_seen': tokens_seen
    }
    torch.save(checkpoint, path)

def evaluate_model(model, train_loader, valid_loader, device, eval_iter, device_type, ptdtype):
    model.eval()
    with torch.no_grad():
        # Wrap evaluation in autocast as well for faster validation steps
        with torch.autocast(device_type = device_type, dtype = ptdtype):
            train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
            valid_loss = calc_loss_loader(valid_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, valid_loss

def generate_print_text(model, tokeniser, device, start_context):
    model.eval()
    # Safely fetch context size from the config instead of embedding weights
    context_size = model.config['context_length'] 
    encoded = text_to_token(start_context, tokeniser).to(device)
    with torch.no_grad():
        tokens = generate_text(model = model, idx = encoded, max_new_tokens = 50, context_size = context_size)
    decoded_text = token_to_text(tokens, tokeniser)
    print(decoded_text.replace('\n', ' '))
    model.train()

class MemmapDataset(Dataset):
    def __init__(self, bin_file_path, context_length):
        # Read the binary file directly from the drive as 16-bit integers
        self.data = np.memmap(bin_file_path, dtype = np.uint16, mode = 'r')
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # Slice exactly what is needed for one sequence plus the target token
        chunk = self.data[idx : idx + self.context_length + 1]
        
        # Convert to int64 tensors for PyTorch embedding lookup
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

def set_optimisers(model, weight_decay, learning_rate, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Any parameter that is 2D or higher  gets weight decay
    decay_params = [p for n, p in param_dict.items() if p.dim() >=  2]
    # Any parameter that is 1D gets zero weight decay
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Fused AdamW runs an optimized C++ kernel, heavily speeding up the update step on AMD/Nvidia GPUs
    use_fused = (device_type == 'cuda')
    
    return torch.optim.AdamW(optim_groups, lr = learning_rate, fused = use_fused)

def get_lr(it, max_iters, warmup_iters, max_lr, min_lr):
    # 1) Linear warmup for a stable start
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # 2) If we push past max_iters, hold at minimum
    if it > max_iters:
        return min_lr
    # 3) Cosine decay down to min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train_model(model, train_loader, valid_loader, optimiser, device, tokeniser):
    max_iters = 100000          # Total parameter updates based on dataset size
    warmup_iters = 2000         # Allow weights to stabilize before high LR
    max_lr = 6e-4
    min_lr = 6e-5
    accumulation_steps = 8      # Number of micro-batches to process before updating weights
    eval_freq = 500
    

    ptdtype = torch.bfloat16 if device == 'cuda' else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled = (ptdtype == torch.float16))

    num_tokens_seen, global_step, update_step = 0, 0, 0
    model.train()
    optimiser.zero_grad(set_to_none = True)

    # Note: train_loader should be pulling from a memory-mapped binary file
    for input_batch, target_batch in train_loader:
        
        # Determine the current Learning Rate and apply it
        lr = get_lr(update_step, max_iters, warmup_iters, max_lr, min_lr)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

        # 1. Forward and Backward pass via Mixed Precision
        with torch.autocast(device_type = device, dtype = ptdtype):
            loss = batch_calc_loss(input_batch, target_batch, model, device)
            # Scale the loss to account for gradient accumulation mathematically
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        num_tokens_seen += input_batch.numel()
        global_step += 1

        # 2. Weight Update Phase (Only triggers after N accumulation steps)
        if global_step % accumulation_steps == 0:
            # Unscale gradients before clipping to ensure correct magnitude checking
            scaler.unscale_(optimiser)
            # Clip gradients to a max norm of 1.0 to prevent explosive divergence
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            
            scaler.step(optimiser)
            scaler.update()
            
            # Flush gradients to free memory
            optimiser.zero_grad(set_to_none = True)
            update_step +=  1

            if update_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model, train_loader, valid_loader, device, 10, device, ptdtype)
                print(f"Step {update_step:06d} | LR {lr:.2e} | "
                      f"Train Loss {train_loss:.3f} | Valid Loss {valid_loss:.3f} | "
                      f"Tokens: {num_tokens_seen / 1e6:.1f}M")
                
                # Highly recommended to save checkpoints dynamically during a 4-day run
                if update_step % (eval_freq * 4) == 0:
                    torch.save(model.state_dict(), f'checkpoint_step_{update_step}.pth')

        if update_step >= max_iters:
            break

    return model

# Test code setup
print(device)
optimiser = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
num_epochs = 1

train_losses, valid_losses, tokens_seen = train_model(
        model = model, train_loader = training_loader, 
        valid_loader = validation_loader, optimiser = optimiser, device = device, 
        num_epochs = num_epochs, eval_freq = 25, eval_iter = 5, 
        start_context = 'Every effort moves you', tokeniser = tokeniser
)
save(model, optimiser, path)