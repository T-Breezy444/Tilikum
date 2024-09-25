import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from transformers import AutoTokenizer
import time
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch import amp 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#-----------------------------------------------------------------------------
#This is set up currently to be trained on one RTX 3060 w/ 12 Gb dedicated ram
block_size = 128
batch_size = 128
max_iters = 10000
learning_rate = 3e-4
eval_iters = 200
n_embd = 128
n_layer = 9
n_head = 16
dropout = 0.2
accumulation_steps = 4
#-----------------------------------------------------------------------------
print('Using device:', device)

chars = ''
with open('C:\\Users\\mango\\dev\\Tilikum\\Models\\BigReddit\\RedditVocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)
print('Vocab size:', vocab_size)

# Tokenizer mappings
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int.get(ch, 0) for ch in s]  # Default to 0 if char not found
decode = lambda x: ''.join([int_to_string.get(i, '?') for i in x])  # Default to '?' if index not found

def get_random_chunk(split):
    filename = (
        'C:\\path\\to\\train.txt'
        if split == 'train'
        else 'C:\\path\\to\\validate.txt'
    )
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            max_start = file_size - (block_size * batch_size + 1)
            if max_start <= 0:
                raise ValueError(
                    f"File {filename} is too small for the given block_size and batch_size."
                )
            start_pos = random.randint(0, max_start)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size + 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', ' ')
            # Encode the decoded block
            encoded = [string_to_int.get(ch, 0) for ch in decoded_block]
            if len(encoded) < block_size * batch_size + 1:
                raise ValueError(
                    "Not enough data read. Adjust block_size or batch_size."
                )
            # Create input (x) and target (y)
            x = torch.tensor(encoded[:-1], dtype=torch.long).view(batch_size, block_size)
            y = torch.tensor(encoded[1:], dtype=torch.long).view(batch_size, block_size)
            return x.to(device), y.to(device)

def get_batch(split):
    return get_random_chunk(split)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # Get the predictions
            logits, _ = self.forward(index_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
m = model.to(device)

# Uncomment and adjust if you need to load a pre-trained model
'''
print('Loading model...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model loaded')
m = model.to(device)
'''


from transformers import get_linear_schedule_with_warmup
#create an optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

num_warmup_steps = 1000
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_iters
)


# Create a learning rate scheduler (for example, StepLR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

# Mixed precision training with adaptive learning rates
scaler = amp.GradScaler()

patience = 5  # Number of iterations to wait for improvement
best_val_loss = float('inf')  # Initialize best validation loss
patience_counter = 0  # Counter for patience

# Training loop
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'Iter {iter}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')

        # Early stopping logic
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']  # Update best validation loss
            patience_counter = 0  # Reset counter if we have an improvement
            print("Validation loss improved. Saving model...")
            # Save the model if needed
            with open('best921.pkl', 'wb') as f:
                pickle.dump(model, f)
        else:
            patience_counter += 1  # Increment counter

        # Check if we should stop training
        if patience_counter >= patience:
            print(f'Early stopping triggered. No improvement for {patience} iterations.')
            break  # Exit the training loop

    # Sample batch of data
    try:
        xb, yb = get_batch('train')
    except ValueError as e:
        print(f"Batch {iter}: {e}")
        continue  # Skip this batch

    # Mixed precision training
    with amp.autocast(device_type='cuda'):
        logits, loss = model(xb, yb)
        loss = loss / accumulation_steps  # Normalize loss for accumulation

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()  # Scale the loss and backward pass

    # Update weights every `accumulation_steps`
    if (iter + 1) % accumulation_steps == 0:
        scaler.step(optimizer)  # Update the optimizer
        scaler.update()  # Update the scale for the next iteration

        # Step the scheduler after optimizer step
        scheduler.step()

print(f'Final loss: {loss.item():.4f}') 

# Save the model
with open('Tilikum-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved!')
