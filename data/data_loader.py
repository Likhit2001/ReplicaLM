import tiktoken
import torch
import os

class DataLoader:
    def __init__(self, B , T):
        self.B = B
        self.T = T
        
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir,"input.txt")
        
        with open(file_path,'r') as f:
            text = f.read()
            
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Total Number of tokens {self.tokens.shape[0]}")
        print(f"Total Number of Batches = {(self.tokens.shape[0]) // (B*T)}")
        
        self.current_position = 0
        
    def next_batch(self):
        B , T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        
        self.current_position += B*T
        
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        
        return x , y
        
        
        
        