import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusers import PixArtAlphaPipeline

class SoundproofAttnProcessor:
    def __init__(self, is_active_mask):
        self.is_active = is_active_mask

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        seq_len = sequence_length
        custom_mask = torch.zeros((seq_len, seq_len), dtype=query.dtype, device=query.device)
        
        anchor_rows = (~self.is_active).unsqueeze(1)
        active_cols = self.is_active.unsqueeze(0)
        custom_mask[anchor_rows & active_cols] = float('-inf')
        
        custom_mask = custom_mask.unsqueeze(0)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=custom_mask
        )

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


print("Loading PixArt...")
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Get tokenizer
tokenizer = pipe.tokenizer

# Encode prompt
prompt = "a red apple on a wooden table"
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=120,
    truncation=True,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids.to("cuda")

# Get the token index for "apple"
tokens = tokenizer.convert_ids_to_tokens(text_input_ids[0])
print("Tokens:", tokens[:15])
apple_idx = None
for i, token in enumerate(tokens):
    if 'apple' in token.lower():
        apple_idx = i
        print(f"Found 'apple' at token index {i}: '{token}'")
        break

if apple_idx is None:
    apple_idx = 3
    print(f"Using default token index {apple_idx}")

# Storage for attention weights
captured_attn = {}

# Hook function to capture cross-attention
def make_cross_attn_hook(block_idx):
    def hook(module, input, output):
        # Try to get attention weights from the module
        if hasattr(module, 'attn_weights'):
            captured_attn[block_idx] = module.attn_weights.detach().cpu()
    return hook

# Attach hooks to cross-attention layers
for i, block in enumerate(pipe.transformer.transformer_blocks):
    if hasattr(block, 'attn2'):
        block.attn2.register_forward_hook(make_cross_attn_hook(i))

print("Running 1 step to capture cross-attention...")
generator = torch.Generator("cuda").manual_seed(42)
_ = pipe(
    "a red apple on a wooden table",
    num_inference_steps=1,
    generator=generator
)

print(f"Captured attention from {len(captured_attn)} blocks")

# Try to get attention from the captured blocks
attn_weights = None
for key, val in captured_attn.items():
    if val is not None:
        attn_weights = val
        print(f"Block {key} attention shape: {val.shape}")
        break

if attn_weights is not None:
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # attn_weights might be [batch, heads, seq_len, text_len] or [batch*heads, seq_len, text_len]
    if attn_weights.dim() == 3:
        num_heads = attn_weights.shape[0]
        seq_len = attn_weights.shape[1]
        text_len = attn_weights.shape[2]
        attn_weights = attn_weights.reshape(1, num_heads, seq_len, text_len)
    
    # Extract attention for "apple" token
    apple_attn = attn_weights[:, :, :, apple_idx]
    print(f"Apple attention shape: {apple_attn.shape}")
    
    # Average across heads
    apple_attn_avg = apple_attn.mean(dim=1)
    print(f"Average apple attention shape: {apple_attn_avg.shape}")
    
    # Handle NaN
    apple_attn_avg = torch.nan_to_num(apple_attn_avg, nan=0.0)
    
    # Normalize to 0-1
    apple_attn_min = apple_attn_avg.min()
    apple_attn_max = apple_attn_avg.max()
    apple_attn_norm = (apple_attn_avg - apple_attn_min) / (apple_attn_max - apple_attn_min + 1e-8)
    
    # Create mask with threshold
    threshold = 0.3
    is_active = (apple_attn_norm > threshold).squeeze(0)
    active_count = is_active.sum().item()
    print(f"Active patches (threshold={threshold}): {active_count} / 1024")
    
    # Show some sample scores
    print(f"Sample attention scores: {apple_attn_norm.squeeze(0)[:10].cpu().numpy()}")
    
    # Now run full inference with the dynamic mask
    soundproof_processor = SoundproofAttnProcessor(is_active_mask=is_active)
    
    for block in pipe.transformer.transformer_blocks:
        if hasattr(block, 'attn1'):
            block.attn1.set_processor(soundproof_processor)
    
    print("Running full inference with dynamic mask...")
    generator = torch.Generator("cuda").manual_seed(42)
    image = pipe(
        "a red apple on a wooden table",
        num_inference_steps=5,
        generator=generator
    ).images[0]
    
    output_path = "/mnt/c/Users/kusha/soundproof_dynamic.png"
    image.save(output_path)
    print(f"Saved {output_path}")
else:
    print("ERROR: No attention weights captured")
