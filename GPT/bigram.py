import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
# ------------
batch_size = 32
block_size = 8
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s): return [stoi[ch] for ch in s]
def decode(l): return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
segmentation_ratio = 0.9
n = int(segmentation_ratio * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices, targets=None):
        # Get some indices, embed them into a tensor
        # with the shape (indices.shape, vocab_size), and return the result
        logits = self.token_embedding_table(indices)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Get an index with the shape (batch, dimension)
        for _ in range(max_new_tokens):
            # Calculate the logits with a shape of (batch, dimension, vocab_size), retain only the last dimension (because of 'BIGRAM'!), and use it to determine the next index
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # becomes (batch, vocab_size)
            probs = F.softmax(logits, dim=1)  # (batch, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, dimension+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# step 29700: train loss 2.4648, val loss 2.4921
# 'we hath nd g t se te y

# I s Ifrithandomy
# Pr Awnch,
# Le n mind at RESThirenes?
# BAD wndone II frouffo t haleer
# LABeishanon.
# KENGLourdr owome d inty nste s ncang, S:
# Lerenos uppotrt: -w youls chr gbou d's HAhr vend:

# Why bury thathou,
# Wheilowilparedend meio head, and INO menctres y! tlou Prin:

# CO, th y e h
# ABe aied ckns thepr honerrakeat I'ssetisoth,
# F y y,
# BR:
# I inoind oromy ve
# Toforto in at ss th ieraknous
# F thetothe S:
# Indes e a ouk mbe pu ll'TII itharand.
# I s, t s mfoun's ols s ouepan d!
# BESA
