from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
from model_files.model_main import GPT
from model_files.config import GPTConfig
from data.data_loader_DDP import DataLoader_DDP
import time
import os
# torchrun --standalone --nproc_per_node=8 train_DDP.py
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


device_type = "cuda" if device.startswith("cuda") else "cpu"


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
# Batch Accumulation
total_batch_size = 524288 // 8
B = 8
T = 1024
# print(f"The ddp world size is {ddp_world_size}")
gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Total no of gradient accumulation needed: {gradient_accumulation_steps}")

print(f"I am GPU ", ddp_rank)

# Input data Loader
training_data_loader = DataLoader_DDP(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size)

# Settong calculation precision
torch.set_float32_matmul_precision('high')

# Model intialization
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())

# torch compile to speed up the training process

model = torch.compile(model)

model.to(device)

# print("Model Weights has been loaded sucessfully")
if ddp:
    model = DDP(model , device_ids =[ddp_local_rank])
    
raw_model = model.module if ddp else model



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
print("complete till here 2.5")
for name , param in raw_model.named_parameters():
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
    loss_accumulation = 0.0
    for micro_step in range(gradient_accumulation_steps):
        x , y = training_data_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps-1)
        # mixed precision
        with torch.autocast(device_type = device_type ,dtype=torch.float32):
            logits, loss = model(x,y)
            loss = loss/gradient_accumulation_steps
            loss_accumulation += loss.detach()
        # so that we don't synchronize the weights which is done in ddp

        loss.backward()
    
    # this is an important step to average out the loss accross all the process 
    if ddp:
        dist.all_reduce(loss_accumulation,op=dist.ReduceOp.AVG)
    
    # normalise the gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    
    lr = lr_schedular(step)
    for params in optimiser.param_groups:
        params['lr'] = lr
    
    optimiser.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    tokens_processed = training_data_loader.B * training_data_loader.T * gradient_accumulation_steps * ddp_world_size
    
    tokens_per_sec = tokens_processed / dt
    
    if master_process:
        print(f"step  {step}  - loss: {loss_accumulation.item():.6f} - norm: {norm:.4f} - lr: {lr:.4f} - dt: {dt*1000:.2f}ms - tok/sec: {tokens_per_sec}")
        
if ddp:
    destroy_process_group()


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


# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# torchrun --standalone --nproc_per_node=2 -m train.train_DDP