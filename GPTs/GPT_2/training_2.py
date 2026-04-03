import torch
from validation_2 import calc_loss_loader, training_loader, validation_loader, batch_calc_loss
from gpt_2 import text_to_token, token_to_text, generate_text, model, device, tokeniser

path = 'GPTs//GPT_2//save.pth'

def save(model, optimiser, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict()
        }, path)
    
def load(model, optimiser, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimiser_state'])

def evaluate_model(model, train_loader, valid_loader, device, eval_iter, device_type, ptdtype):
    model.eval()
    with torch.no_grad():
        # Wrap evaluation in autocast as well for faster validation steps
        with torch.autocast(device_type=device_type, dtype=ptdtype):
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            valid_loss = calc_loss_loader(valid_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, valid_loss

def generate_print_text(model, tokeniser, device, start_context):
    model.eval()
    # Safely fetch context size from the config instead of embedding weights
    context_size = model.config['context_length'] 
    encoded = text_to_token(start_context, tokeniser).to(device)
    with torch.no_grad():
        tokens = generate_text(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_to_text(tokens, tokeniser)
    print(decoded_text.replace('\n', ' '))
    model.train()

def train_model(model, train_loader, valid_loader, optimiser, device, num_epochs, eval_freq, eval_iter, start_context, tokeniser):
    train_losses, valid_losses, tokens_seen = [], [], []
    num_tokens_seen, global_step = 0, -1

    # AMP SETUP: 
    # PyTorch uses 'cuda' as the device type for both NVIDIA and AMD ROCm under the hood
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    # We prefer bfloat16 if your GPU supports it (RDNA architecture usually does).
    # It has the same range as float32, preventing gradient underflow/overflow.
    ptdtype = torch.bfloat16 if (device_type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    
    # GradScaler is only needed if we fallback to float16. It safely scales gradients to prevent them from vanishing.
    scaler = torch.amp.GradScaler('cuda', enabled=(ptdtype == torch.float16))

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimiser.zero_grad()
            
            # 1. Forward pass under mixed precision context
            with torch.autocast(device_type=device_type, dtype=ptdtype):
                loss = batch_calc_loss(input_batch, target_batch, model, device)
            
            # 2. Backward pass with scaled loss
            scaler.scale(loss).backward()
            
            # 3. Optimizer step through the scaler
            scaler.step(optimiser)
            
            # 4. Update the scaler for the next iteration
            scaler.update()

            num_tokens_seen += input_batch.numel() # More accurate than += 1
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model, train_loader, valid_loader, device, eval_iter, device_type, ptdtype)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                tokens_seen.append(num_tokens_seen)
                print(f'Ep {epoch + 1} (step {global_step:06d}) | ' 
                      f'Train loss {train_loss:.3f} | ' 
                      f'Valid loss {valid_loss:.3f}'
                    )
                
        generate_print_text(model, tokeniser, device, start_context)

    return train_losses, valid_losses, tokens_seen

# Test code setup
if __name__ == '__main__':
    print(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 1

    train_losses, valid_losses, tokens_seen = train_model(
        model=model, train_loader=training_loader, 
        valid_loader=validation_loader, optimiser=optimiser, device=device, 
        num_epochs=num_epochs, eval_freq=25, eval_iter=5, 
        start_context='Every effort moves you', tokeniser=tokeniser
    )
    save(model, optimiser, path)