import torch
import config
import tiktoken


class DataHandler():
    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        self.enc = tiktoken.get_encoding("gpt2")
        self.data = torch.tensor(self.enc.encode(text), dtype=torch.long)

    def get_batch(self):
        ix = torch.randint(len(self.data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
        x = torch.stack([self.data[i:i+config.BLOCK_SIZE] for i in ix])
        y = torch.stack([self.data[i+1:i+config.BLOCK_SIZE+1] for i in ix])
        
        return x, y

    def get_vocab_size(self):
        return self.enc.n_vocab