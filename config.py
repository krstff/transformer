# Standard GPT2 hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 256      # Maximum context length (sequence length)
EMBED_SIZE = 512      # Dimensionality of the embeddings
NUM_HEADS = 8         # Number of attention heads
NUM_LAYERS = 10       # Number of transformer blocks
ATTENTION_DROP = 0.0
RESID_DROP = 0.0

# Finetuning
LEARNING_RATE = 5e-5
STEPS = 3_000
