from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
from model_files.model_main import GPT
from model_files.config import GPTConfig
from data.data_loader import DataLoader


# Device detection
# ===
device ="cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
device ="cpu"
print(f"Using the {device}")

# Input data Loader
training_data_loader = DataLoader(4,32)

# Model intialization
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
print("Model Weights has been loaded sucessfully")
model.to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    
    optimiser.zero_grad()
    x , y = training_data_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x,y)
    loss.backward()
    optimiser.step()
    print(f"step {i}, loss: {loss.item()}")


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
