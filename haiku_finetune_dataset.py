from datasets import load_dataset
import tiktoken
import pyarrow as pa
import lance

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc.eot_token # This is usually 50256

ds = load_dataset("statworx/haiku", split="train")

text_column = 'text'
all_tokens = []

print("Tokenizing Haikus...")
for row in ds:
    haiku_text = row[text_column]
    tokens = enc.encode(haiku_text)
    
    # append the eot token so the model knows when the haiku is over
    tokens.append(EOT_TOKEN)
    
    all_tokens.extend(tokens)

print(f"Total tokens extracted: {len(all_tokens):,}")

token_array = pa.array(all_tokens, type=pa.int64())
table = pa.Table.from_arrays([token_array], names=['input_ids'])

lance.write_dataset(table, "data/haiku_dataset.lance", mode="overwrite")