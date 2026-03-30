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

class LanceDataHandler:
    def __init__(self, path, tokenizer, max_tokens=5_000_000):
        self.enc = tokenizer        
        self.ds = lance.dataset(path)
        print(self.ds.schema)
        
        string_cols = [f.name for f in self.ds.schema if 'utf8' in str(f.type).lower() or 'string' in str(f.type).lower()]
        
        if not string_cols:
            raise ValueError("Could not find any text columns in this Lance dataset!")
            
        # Prioritize common names if they exist, otherwise just grab the first string column
        if 'text' in string_cols: self.col_name = 'text'
        elif 'content' in string_cols: self.col_name = 'content'
        else: self.col_name = string_cols[0]
            
        print(f"Auto-detected text column: '{self.col_name}'")
        print("Streaming and tokenizing data in batches...")
        
        tokens = []
        
        for batch in self.ds.scanner(columns=[self.col_name]).to_batches():
            text_list = batch[self.col_name].to_pylist()
            
            chunk_text = " \n\n ".join(text_list)
            tokens.extend(self.enc.encode(chunk_text))
            
            if len(tokens) >= max_tokens:
                print(f"Reached {max_tokens:,} token limit. Stopping extraction.")
                break

        self.data = torch.tensor(tokens, dtype=torch.long)
        print(f"Dataset successfully loaded! Total tokens ready for training: {len(self.data):,}")

    def get_batch(self):
        ix = torch.randint(len(self.data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
        x = torch.stack([self.data[i:i+config.BLOCK_SIZE] for i in ix])
        y = torch.stack([self.data[i+1:i+config.BLOCK_SIZE+1] for i in ix])
        
        return x, y

    def get_vocab_size(self):
        return self.enc.n_vocab