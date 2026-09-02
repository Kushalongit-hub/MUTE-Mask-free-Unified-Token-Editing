import torch
import torch.nn.functional as F

B = 1
H = 8
SEQ_LEN = 1024
D_HEAD = 64

Q = torch.randn(B, H, SEQ_LEN, D_HEAD, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, SEQ_LEN, D_HEAD, dtype=torch.float16, device="cuda")
V = torch.randn(B, H, SEQ_LEN, D_HEAD, dtype=torch.float16, device="cuda")

is_active = torch.zeros(SEQ_LEN, dtype=torch.bool, device="cuda")
is_active[400:450] = True

anchor_mask = ~is_active
active_mask = is_active

attn_mask = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.float16, device="cuda")
anchor_indices = torch.where(anchor_mask)[0]
active_indices = torch.where(active_mask)[0]
attn_mask[anchor_indices[:, None], active_indices[None, :]] = float('-inf')

attn_mask = attn_mask.reshape(1, 1, SEQ_LEN, SEQ_LEN)

output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)

print("Output shape:", output.shape)
print("Mask at (anchor=0, active=400):", attn_mask[0, 0, 0, 400].item())
print("Mask at (active=400, anchor=0):", attn_mask[0, 0, 400, 0].item())
