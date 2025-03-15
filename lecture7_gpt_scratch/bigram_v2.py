import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
# increasing the number of iterations as learning rate is lower
max_iters = 5000
eval_interval = 300
# as self attention can make very high lr so reducing the lr here
learning_rate = 1e-3
# run on a gpu if u hav it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#no of embeding dimensions
n_embd= 32
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # as device is cuda , when we load the data we move it to device
    x, y = x.to(device), y.to(device)
    return x, y

# as here we are just calculating the loss and no back propogation is done 
# so we set no_grad  goood practice
@torch.no_grad()
# this function averages up the loss over multiple batches
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # here for both split train and val we can get the loss avg
            # in batches which in total runs eval_iter times
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # right now the network will behave same in train and val mode
    # as we hav no dropout layers no batch turn layers buts its good practice 
    # to keep the mode intact which is training right now
    model.train()
    return out

# single head self attention
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        #key , querry and value linear layers
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # as trill is not a parameter in pytorch so we use register_buffer to make it
        # this is the lower tringualar matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

#Multihead attention
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # we crate multiple head in a list
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run these multiple heads in parallel into a list and concatinate it to the output
        # concatinating over the chanel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # we embed the identity of these tokens
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        #We ebmd the positions of these tokens each pos frm block size to block_size -1 has its own embedding
        self.position_embedding_table= nn.Embedding(block_size,n_embd)
        # calling the self attention single head class
        #self.sa_head= Head(n_embd)
        # we tried using self attention it reduced  the loss but not too much so now we use multi head attention
        # from each communication chanel we get vectors and we hav 4 of chanels
        # that gives us 4* 8= 32 vectors
        self.sa_heads= MultiHeadAttention(4, int(n_embd/4)) #i.e 4 heas of 8 - dimensional self-attention
        # set up a layer for the embeddings
        self.lm_head= nn.Linear(n_embd, vocab_size)
        

    def forward(self, idx, targets=None):
        B,T= idx.shape

        # idx and targets are both (B,T) tensor of integers
        # now as we r using vocab_size , n_embd it will give us token embedding
        token_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        # pos embedding from 0 to T-1
        pos_emb= self.position_embedding_table(torch.arange(T,device= device)) # (T,C)
        # as both embedding are embedding of the inputs based on attention 
        # x has token identities and position in which the tokens occur
        x= token_emb+pos_emb
        # once we use token embedding and pos embedding in x we pass that to self attention
        # so that it undertands the contex better
        #based on tranformer architeture
        #x=self.sa_head(x)
        # we use multihead elf attention
        x=self.sa_heads(x)
        logits= self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                # becos we r using pos embedding now
                # so that it doesnt go out of scope we never pass more than
                # block sie elements
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx
model = BigramLanguageModel()
# when we create the model e move its parameters to device
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        # here when we call the estimate loss we can get pretty accurate train and 
        # val loss
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
# also the context to generate shd be in the device cuda
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))