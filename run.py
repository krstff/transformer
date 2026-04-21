from train import Trainer
import torch, tiktoken
from model import GPT2
import config
import argparse

def train(filename, tokenizer, output_path):
    trainer = Trainer(filename, tokenizer, True, output_path)
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
    parser = argparse.ArgumentParser()
    
    # Subcommands or Flags
    parser.add_argument("--mode", type=str, choices=["train", "gen"], required=True, 
                        help="Choose 'train' to train the model or 'gen' to generate text.")
    
    # Arguments for Generation
    parser.add_argument("--prompt", type=str, default="Once upon a time", 
                        help="The starting text for generation.")
    parser.add_argument("--tokens", type=int, default=50, 
                        help="Number of tokens to generate.")
    parser.add_argument("--weights", type=str, default="data/my_model_openweb.pth", 
                        help="Path to the saved model weights.")
    parser.add_argument("--times", type=int, default=1, 
                        help="Number of times to run the generation prompt.")
    
    # Arguments for Training
    parser.add_argument("--data", type=str, default="data/openwebtext_1M.lance/", 
                        help="Path to the training data.")
    parser.add_argument("--output", type=str, default="data/model.pth", 
                        help="Path to save the trained model weights.")

    args = parser.parse_args()
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.mode == "train":
        print(f"Starting training using data: {args.data}")
        train(args.data, tokenizer, args.output)
        
    elif args.mode == "gen":
        print(f"Loading model from {args.weights}...")
        model = load_model(tokenizer.n_vocab, args.weights)
        for i in range(args.times):
            generate(model, tokenizer, prompt=args.prompt, max_tokens=args.tokens)

