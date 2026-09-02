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


class CrossAttnStoreProcessor:
    """Stores cross-attention weights for dynamic mask extraction."""
    def __init__(self):
        self.attn_weights = None
        self.q = None
        self.k = None
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        self.q = query.detach()
        self.k = key.detach()
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention scores manually
        head_dim = query.shape[-1]
        scale = head_dim ** -0.5
        attn_weights = torch.bmm(query, key.transpose(-2, -1)) * scale
        
        # Check for NaN
        has_nan = torch.isnan(attn_weights).any().item()
        has_inf = torch.isinf(attn_weights).any().item()
        print(f"  attn_weights has_nan={has_nan}, has_inf={has_inf}, min={attn_weights.min().item():.4f}, max={attn_weights.max().item():.4f}")
        
        # Store a clone to avoid issues with autograd
        self.attn_weights = attn_weights.clone().detach()
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        hidden_states = torch.bmm(attn_weights, value)
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

# Create cross-attention store processor
store_processor = CrossAttnStoreProcessor()

# Attach to cross-attention layers
for block in pipe.transformer.transformer_blocks:
    if hasattr(block, 'attn2'):
        block.attn2.set_processor(store_processor)

print("Running 1 step to capture cross-attention...")
generator = torch.Generator("cuda").manual_seed(42)
_ = pipe(
    "a red apple on a wooden table",
    num_inference_steps=1,
    generator=generator
)

# Get stored attention weights
if store_processor.attn_weights is not None:
    attn_weights = store_processor.attn_weights
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Q shape: {store_processor.q.shape if store_processor.q is not None else 'None'}")
    print(f"K shape: {store_processor.k.shape if store_processor.k is not None else 'None'}")
    
    # attn_weights shape: [batch*heads, seq_len, text_len]
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
    denom = apple_attn_max - apple_attn_min
    if denom < 1e-8:
        print(f"Warning: attention range too small ({denom.item():.6f}), using uniform mask")
        is_active = torch.zeros_like(apple_attn_avg.squeeze(0), dtype=torch.bool)
        is_active[:100] = True
    else:
        apple_attn_norm = (apple_attn_avg - apple_attn_min) / (denom + 1e-8)
        
        # Create mask with threshold
        threshold = 0.3
        is_active = (apple_attn_norm > threshold).squeeze(0)
    
    active_count = is_active.sum().item()
    print(f"Active patches: {active_count} / 1024")
    
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
