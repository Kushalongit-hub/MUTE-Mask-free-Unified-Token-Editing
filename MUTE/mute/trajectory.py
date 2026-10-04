"""Post-scheduler source latent restoration, separate from attention masking."""
import torch
import torch.nn.functional as F


class SourceTrajectory:
    def __init__(self):
        self.latents = []
        self.timesteps = []

    def record(self, step, timestep, latents):
        if step != len(self.latents) or not torch.isfinite(latents).all():
            raise ValueError("Source trajectory must be finite and recorded sequentially")
        self.timesteps.append(float(timestep))
        self.latents.append(latents.detach().cpu().clone())

    def restore(self, step, timestep, latents, active, start=0, end=None):
        if step >= len(self.latents) or float(timestep) != self.timesteps[step]:
            raise ValueError("Target schedule differs from source trajectory")
        reference = self.latents[step].to(device=latents.device, dtype=latents.dtype)
        if reference.shape != latents.shape:
            raise ValueError("Source and target latent shapes differ")
        if step < start or (end is not None and step >= end):
            return
        weight = latent_edit_weight(active, latents)
        # Exact endpoints and unchanged-source identity; interpolate only the ring.
        blended = reference + (latents - reference) * weight
        latents.copy_(torch.where(weight == 1, latents, torch.where(weight == 0, reference, blended)))


def edit_patch_weights(active, margin=0, feather=0):
    if margin < 0 or feather < 0:
        raise ValueError("Mask margin and feather must be nonnegative")
    active = torch.as_tensor(active, dtype=torch.bool)
    side = int(active.numel() ** 0.5)
    if active.ndim != 1 or side == 0 or side * side != active.numel():
        raise ValueError("Expected square patch mask")
    core = active.reshape(1, 1, side, side).float()
    if margin:
        core = F.max_pool2d(core, 2 * margin + 1, stride=1, padding=margin)
    weight = core.clone()
    for distance in range(1, feather + 1):
        ring = F.max_pool2d(core, 2 * distance + 1, stride=1, padding=distance)
        weight = torch.maximum(weight, ring * (1 - distance / (feather + 1)))
    return weight.flatten()


def latent_edit_weight(weight, latents):
    weight = torch.as_tensor(weight, device=latents.device, dtype=latents.dtype)
    side = int(weight.numel() ** 0.5)
    if weight.ndim != 1 or side == 0 or side * side != weight.numel() or latents.ndim != 4:
        raise ValueError("Expected square edit weights and BCHW latents")
    if not torch.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
        raise ValueError("Edit weights must be finite and in [0,1]")
    return F.interpolate(weight.reshape(1, 1, side, side), size=latents.shape[-2:], mode="nearest")


def latent_active_mask(active, latents):
    active = torch.as_tensor(active, dtype=torch.bool, device=latents.device)
    side = int(active.numel() ** 0.5)
    if active.ndim != 1 or side * side != active.numel() or latents.ndim != 4:
        raise ValueError("Expected square patch mask and BCHW latents")
    return F.interpolate(active.reshape(1, 1, side, side).float(),
                         size=latents.shape[-2:], mode="nearest").bool()
