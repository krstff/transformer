import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import DataHandler, LanceDataset, LanceSampler
from torch.utils.data import DataLoader
from model import GPT2
from dataset import DataHandler
import config
import torch.nn as nn


class Trainer():
    def __init__(self, filename, tokenizer, is_lance=False):
        if is_lance:
            # Assume filename is the path to the Lance dataset
            self.dataset = LanceDataset(filename, block_size=config.BLOCK_SIZE)
            self.sampler = LanceSampler(self.dataset, block_size=config.BLOCK_SIZE)
            self.dataloader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE, sampler=self.sampler, shuffle=False)
            self.vocab_size = tokenizer.n_vocab
        else:
            self.data_handler = DataHandler(filename, tokenizer)
            self.vocab_size = self.data_handler.get_vocab_size()


    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = GPT2(self.vocab_size).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        dataloader_iter = iter(self.dataloader) if hasattr(self, 'dataloader') else None


        # Track loss for matplotlib
        tracked_losses = []
        tracked_steps = []
        step = 0

        print("Starting training...")
        while step < config.STEPS:
            # using lance loader
            if hasattr(self, 'dataloader'):
                try:
                    batch = next(dataloader_iter)
                    xb = batch['input_ids'].to(device)
                    yb = batch['labels'].to(device)
                except StopIteration:
                    # Restart dataloader for multiple epochs if needed
                    dataloader_iter = iter(self.dataloader)
                    batch = next(dataloader_iter)
                    xb = batch['input_ids'].to(device)
                    yb = batch['labels'].to(device)
            else:
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

            step += 1

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
