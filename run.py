from train import Trainer
import torch, tiktoken
from model import GPT2
import config

def train(filename, tokenizer):
    trainer = Trainer(filename, tokenizer)
    trainer.train()

def load_model(vocab_size, filename) -> GPT2:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2(vocab_size).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval() 
    return model

def generate(model, tokenizer, prompt="", max_tokens=50):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    context_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_tensor = model.generate(context_tensor, max_new_tokens=max_tokens)
    generated_ids = generated_tensor.squeeze(0).tolist()
    output_text = tokenizer.decode(generated_ids)
    
    print(output_text)
    
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    # model = load_model(tokenizer.n_vocab, 'data/my_model.pth')
    # generate(model, tokenizer, "Romeo looked at")
    train('/data/openwebtext_1M.lance/', tokenizer)

