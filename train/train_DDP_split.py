from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
from model_files.model_main import GPT
from model_files.config import GPTConfig
from data.data_loader_DDP_split import DataLoader_DDP
import time
import os
from test.heallswag import render_example, iterate_examples

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


ddp = int(os.environ.get('RANK', -1)) != -1 

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
training_data_loader = DataLoader_DDP(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split="train")

val_data_loader = DataLoader_DDP(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split="val")


# Settong calculation precision
torch.set_float32_matmul_precision('high')

# Model intialization
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())

# torch compile to speed up the training process

# model = torch.compile(model)

model.to(device)

# print("Model Weights has been loaded sucessfully")
if ddp:
    model = DDP(model , device_ids =[ddp_local_rank])
    
raw_model = model.module if ddp else model



# lr schedular
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 120
max_steps = 3528

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


log_dir = "log"
os.makedirs(log_dir,exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

# ======== helper ===== #
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# train loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # Validation loss logging
    if step % 25 == 0 or last_step :
        model.eval()
        val_data_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x , y = val_data_loader.next_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type = device_type ,dtype=torch.float32):
                    logits, loss = model(x,y)
                loss = loss/val_loss_steps
                val_loss_accum += loss.detach()
    
        if ddp:
            dist.all_reduce(val_loss_accum,op=dist.ReduceOp.AVG)
        
        if master_process:
            print(f"validation_loss : {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                
    
    if (step % 25 == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
        
    
    model.train()
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
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accumulation.item():.6f}\n")
        
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
# torchrun --standalone --nproc_per_node=2 -m train.train_DDP_split