import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GPT2
from dataset import DataHandler
import config
import torch.nn as nn


class Trainer():
    def __init__(self, filename, tokenizer, is_lance=False, is_pretokenized=False):
        self.data_handler = DataHandler(filename, tokenizer)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = GPT2(self.data_handler.get_vocab_size()).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        # Track loss for matplotlib
        tracked_losses = []
        tracked_steps = []

        print("Starting training...")
        for step in range(config.STEPS):
            xb, yb = self.data_handler.get_batch()
            xb, yb = xb.to(device), yb.to(device)

            # Decrease the precision to speed up training and allow for bigger batch size
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(xb)
                
                B, T, C = logits.shape
                logits_flat = logits.view(B*T, C)
                targets_flat = yb.view(B*T)
                loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                tracked_losses.append(loss.item())
                tracked_steps.append(step)

            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), 'data/my_model_openweb2.pth')

        # matplotlib save training_loss_graph
        plt.figure(figsize=(10, 5))
        plt.plot(tracked_steps, tracked_losses, label='Training Loss', color='blue')
        plt.title('Model Training Loss Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Cross Entropy Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('training_loss_graph.png')
