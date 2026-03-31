import torch
import config
import tiktoken
import os
import glob
import lance


class DataHandler():
    def __init__(self, path, tokenizer):
        self.enc = tokenizer
        text = ""

        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, '*.txt'))
            print(f"Found {len(files)} files in directory. Loading into memory...")
            
            # !! Loads the entire dataset into RAM
            for f_name in files:
                with open(f_name, 'r', encoding='utf-8') as f:
                    text += f.read() + " \n\n " 
                    
        elif os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

        print("Tokenizing data...")
        self.data = torch.tensor(self.enc.encode(text), dtype=torch.long)
        print(f"Dataset loaded successfully! Total tokens: {len(self.data):,}")

    def get_batch(self):
        ix = torch.randint(len(self.data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
        x = torch.stack([self.data[i:i+config.BLOCK_SIZE] for i in ix])
        y = torch.stack([self.data[i+1:i+config.BLOCK_SIZE+1] for i in ix])
        
        return x, y

    def get_vocab_size(self):
        return self.enc.n_vocab
