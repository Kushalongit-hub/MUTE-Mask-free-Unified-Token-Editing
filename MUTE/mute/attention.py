"""PixArt-specific wrappers that preserve the installed attention processor."""
from __future__ import annotations

import torch


def directional_bias(active, mode, dtype, device, penalty=float("inf")):
    if penalty <= 0 or penalty != penalty:
        raise ValueError("Attention penalty must be positive")
    active = torch.as_tensor(active, dtype=torch.bool, device=device)
    if active.ndim != 1 or not active.any() or active.all():
        raise ValueError("Localization must contain both active and background patches")
    background = ~active
    blocked = background[:, None] & active[None, :]
    if mode == "symmetric":
        blocked |= active[:, None] & background[None, :]
    elif mode == "reverse":
        blocked = active[:, None] & background[None, :]
    elif mode != "directional":
        raise ValueError(f"Unknown intervention: {mode}")
    return torch.zeros(blocked.shape, dtype=dtype, device=device).masked_fill(blocked, -penalty)


class DirectionalProcessor:
    def __init__(self, original, active, mode="directional", start_step=0, end_step=None, penalty=float("inf")):
        if start_step < 0 or (end_step is not None and end_step <= start_step):
            raise ValueError("Invalid intervention interval")
        self.original, self.active, self.mode = original, active, mode
        self.start_step, self.end_step, self.penalty = start_step, end_step, penalty
        self.step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        step = self.step
        self.step += 1
        if self.mode == "none" or step < self.start_step or (self.end_step is not None and step >= self.end_step):
            return self.original(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        if encoder_hidden_states is not None or hidden_states.ndim != 3:
            raise ValueError("DirectionalProcessor supports PixArt sequence self-attention only")
        batch, length, _ = hidden_states.shape
        bias = directional_bias(self.active, self.mode, hidden_states.dtype, hidden_states.device, self.penalty)
        if bias.shape != (length, length):
            raise ValueError("Localization length differs from transformer patch count")
        # Match AttnProcessor2_0's expected [batch*heads, query, key] input.
        bias = bias.unsqueeze(0).expand(batch * attn.heads, -1, -1)
        if attention_mask is not None:
            existing = attn.prepare_attention_mask(attention_mask, length, batch)
            if existing.dtype == torch.bool:
                existing = torch.zeros_like(existing, dtype=bias.dtype).masked_fill(~existing, -torch.inf)
            bias = bias + existing
        return self.original(attn, hidden_states, attention_mask=bias, **kwargs)


class AttentionStore:
    def __init__(self, token_indices, cfg=True, capture_steps=(5, 10, 15)):
        self.token_indices = tuple(token_indices)
        self.cfg = cfg
        self.capture_steps = set(capture_steps)
        self.maps = []
        self.records = []

    def add(self, attn, hidden, context, mask, layer, step):
        if step not in self.capture_steps:
            return
        if hidden.ndim != 3 or attn.spatial_norm is not None or attn.group_norm is not None:
            raise ValueError("Capture supports standard PixArt sequence attention only")
        batch, patches, _ = hidden.shape
        if self.cfg and batch != 2:
            raise ValueError("Capture expects one image with negative/positive CFG batches")
        if not self.cfg and batch != 1:
            raise ValueError("Capture expects a single image")
        if attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)
        q, k = attn.to_q(hidden), attn.to_k(context)
        head_dim = k.shape[-1] // attn.heads
        q = q.view(batch, patches, attn.heads, head_dim).transpose(1, 2)
        k = k.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        # Diffusers PixArt orders negative first, positive second.
        selected = 1 if self.cfg else 0
        scores = (q[selected].float() @ k[selected].float().transpose(-1, -2)) * head_dim ** -0.5
        if mask is not None:
            prepared = attn.prepare_attention_mask(mask, context.shape[1], batch)
            prepared = prepared.view(batch, attn.heads, -1, context.shape[1])[selected]
            if prepared.dtype == torch.bool:
                scores = scores.masked_fill(~prepared, -torch.inf)
            else:
                scores = scores + prepared.float()
        probabilities = scores.softmax(-1)
        values = probabilities[..., list(self.token_indices)].sum(-1).mean(0)
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite cross-attention localization")
        self.maps.append(values.detach().cpu())
        self.records.append({"layer": layer, "step": step})

    def localize(self, threshold=0.3):
        if not 0 < threshold < 1 or not self.maps:
            raise ValueError("Need captured maps and a threshold strictly between zero and one")
        mean = torch.stack(self.maps).mean(0)
        span = mean.max() - mean.min()
        if span <= 1e-8:
            raise ValueError("Degenerate localization; no arbitrary fallback mask is allowed")
        normalized = (mean - mean.min()) / span
        active = normalized > threshold
        if not active.any() or active.all():
            raise ValueError("Localization has no usable foreground/background split")
        return normalized, active


class CaptureProcessor:
    def __init__(self, original, store, layer):
        self.original, self.store, self.layer = original, store, layer
        self.step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is None:
            raise ValueError("CaptureProcessor must be attached to cross-attention")
        with torch.no_grad():
            self.store.add(attn, hidden_states, encoder_hidden_states, attention_mask, self.layer, self.step)
        self.step += 1
        return self.original(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
