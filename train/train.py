from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
from model_files.model_main import GPT
from model_files.config import GPTConfig
from data.data_loader import DataLoader
import time

# Device detection
# ===
device ="cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
# device ="cpu"
print(f"Using the {device}")

# Input data Loader
training_data_loader = DataLoader(8,1024)

# Settong calculation precision
torch.set_float32_matmul_precision('high')

# Model intialization
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())

# torch compile to speed up the training process
model = torch.compile(model)
print("Model Weights has been loaded sucessfully")
model.to(device)

# lr schedular
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def lr_schedular(it):
    
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr-min_lr)
    

# Optimizer
# optimiser = torch.optim.AdamW(model.parameters(), betas=(0.9,0.95), eps=1e-8)

# Adding weight decay
decay = []
decay_not = []

for name , param in model.named_parameters():
    if param.ndim >= 2:
        decay.append(param)
    else:
        decay_not.append(param)
        
modified_params_with_decay = [{"params" : decay , "weight_decay": 0.1}, {"params": decay_not, "weight_decay": 0.0}]

optimiser = torch.optim.AdamW(modified_params_with_decay, lr=max_lr , betas=(0.9,0.95), eps=1e-8)

# train loop
for step in range(max_steps):
    t0 = time.time()
    optimiser.zero_grad()
    x , y = training_data_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    
    # mixed precision
    with torch.autocast(device_type = device,dtype=torch.float32):
        logits, loss = model(x,y)
        
    loss.backward()
    
    # normalise the gradients
    norm = torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
    
    lr = lr_schedular(step)
    for params in optimiser.param_groups:
        params['lr'] = lr
    
    optimiser.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    tokens_processed = training_data_loader.B * training_data_loader.T
    tokens_per_sec = tokens_processed / dt
    
    print(f"step  {step}  - loss: {loss.item():.6f} - norm: {norm:.4f} - lr: {lr:.4f} - dt: {dt*1000:.2f}ms - tok/sec: {tokens_per_sec}")


import sys
sys.exit(0)

num_return_sentences = 5
max_length = 30
model.eval()
model.to(device)

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sentences, 1) # 5 ,8
x = tokens
x = x.to(device)

torch.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=1) # (B, vocab_size)
        
        topk_probs , topk_indices = torch.topk(probs,50,dim= -1)
        
        ix = torch.multinomial(topk_probs,1)
        
        xcol = torch.gather(topk_indices, -1,ix)
        x = torch.cat((x,xcol),dim=1)

for i in range(num_return_sentences):
    tokens = x[i,:max_length].tolist()
    decode = enc.decode(tokens)
    print(">", decode)
