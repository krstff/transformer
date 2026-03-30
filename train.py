import torch
import torch.nn.functional as F
from model import GPT2
from dataset import DataHandler, LanceDataHandler, PreTokenizedLanceDataHandler
import config
import torch.nn as nn


class Trainer():
    def __init__(self, filename, tokenizer, is_lance=False, is_pretokenized=False):
        if is_pretokenized:
            self.data_handler = PreTokenizedLanceDataHandler(filename, tokenizer)
        elif is_lance:
            self.data_handler = LanceDataHandler(filename, tokenizer)
        else:
            self.data_handler = DataHandler(filename, tokenizer)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = GPT2(self.data_handler.get_vocab_size()).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        print("Starting training...")
        for step in range(5000):
            xb, yb = self.data_handler.get_batch()
            xb, yb = xb.to(device), yb.to(device)
            
            logits = model(xb)
            
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = yb.view(B*T)
            
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), 'data/my_model.pth')
