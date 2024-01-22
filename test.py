import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer


vocab_size = 4096 # the Llama 2 tokenizer has 32K tokens
# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
hidden_dim = 4 * dim
hidden_dim = int(2 * hidden_dim / 3)
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
head_dim = dim // n_heads
norm_eps = 1e-5 

# Define your model arguments here. Replace with actual values.
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    norm_eps=norm_eps,
    head_dim=head_dim,
    hidden_dim=hidden_dim,
)
tokenizer = Tokenizer("data/tok4096.model")
# Initialize the model
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
model.to('cuda')
# Load the checkpoint
checkpoint = torch.load("out/ckpt.pt", map_location=torch.device('cuda'))
state_dict = checkpoint["model"]

unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# Prepare the input data
# This is a placeholder, replace with your actual data preparation
prompt = "Princess Eliya"
input_ids = torch.tensor([tokenizer.encode(prompt)]).flatten().to('cuda')
seqlens = [len(input_ids)]

# Run inference
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(input_ids, seqlens)

output_text = tokenizer.decode(output.argmax(dim=-1).tolist())
# Process the output as needed
print(output_text)
