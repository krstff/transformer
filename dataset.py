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


# Sample lance loader https://lance.org/examples/python/llm_training/#imports-and-setup
def from_indices(dataset, indices):
    """Load the elements on given indices from the dataset"""
    chunk = dataset.take(indices).to_pylist()
    chunk = list(map(lambda x: x['input_ids'], chunk))
    return chunk

class LanceDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        block_size,
    ):
        # Load the lance dataset from the saved path
        self.ds = lance.dataset(dataset_path)
        self.block_size = block_size

        # Doing this so the sampler never asks for an index at the end of text
        self.length = self.ds.count_rows() - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Generate a window of indices starting from the current idx to idx+block_size
        and return the tokens at those indices
        """
        window = np.arange(idx, idx + self.block_size)
        sample = from_indices(self.ds, window)

        return {"input_ids": torch.tensor(sample), "labels": torch.tensor(sample)}

class LanceSampler(Sampler):
    r"""Samples tokens randomly but `block_size` indices apart.

    Args:
        data_source (Dataset): dataset to sample from
        block_size (int): minimum index distance between each random sample
    """

    def __init__(self, data_source, block_size=512):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.available_indices = list(range(0, self.num_samples, block_size))
        np.random.shuffle(self.available_indices)

    def __iter__(self):
        yield from self.available_indices

    def __len__(self) -> int:
        return len(self.available_indices)
