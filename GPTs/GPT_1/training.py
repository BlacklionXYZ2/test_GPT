import torch, torch.nn as nn
from training_validation import calc_loss_loader, model, device, tokeniser, training_loader, validation_loader, batch_calc_loss
from gpt_1 import text_to_token, token_to_text, generate_text
# \

path = 'python//EPQ_projects//GPTs//GPT_1//save.pth'

def save(model, optimiser, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict()
        }, path)
    
def load(model, optimiser, path):
    checkpoint = torch.load(path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimiser_state'])


def evaluate_model(model, train_loader, valid_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        valid_loss = calc_loss_loader(valid_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, valid_loss


def generate_print_text(model, tokeniser, device, start_context):
    model.eval()
    context_size = model.pos_embed.weight.shape[0]
    encoded = text_to_token(start_context, tokeniser).to(device)
    with torch.no_grad():
        tokens = generate_text(model = model, idx = encoded, max_new_tokens = 50, context_size = context_size)
    decoded_text = token_to_text(tokens, tokeniser)
    print(decoded_text.replace('\n', ' '))
    model.train()


def train_model(model, train_loader, valid_loader, optimiser, device, num_epochs, eval_freq, eval_iter, start_context, tokeniser):
    train_losses, valid_losses, tokens_seen = [], [], []
    num_tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:
            optimiser.zero_grad()
            loss = batch_calc_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimiser.step()

            num_tokens_seen += 1
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model, train_loader, valid_loader, device, eval_iter)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                tokens_seen.append(num_tokens_seen)
                print(f'Ep {epoch + 1} (step {global_step:06d})' 
                      f'Train loss {train_loss:.3f}' 
                      f'Valid loss {valid_loss:.3f}'
                    )
                
        generate_print_text(model, tokeniser, device, start_context)

    return train_losses, valid_losses, tokens_seen


#test code
#torch.manual_seed(123)
optimiser = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
num_epochs = 3
load(model, optimiser, path)
train_losses, valid_losses, tokens_seen = train_model(
    model = model, train_loader = training_loader, 
    valid_loader = validation_loader, optimiser = optimiser, device = device, 
    num_epochs = num_epochs, eval_freq = 25, eval_iter = 5, 
    start_context = 'Every effort moves you', tokeniser = tokeniser
    )
save(model, optimiser, path)