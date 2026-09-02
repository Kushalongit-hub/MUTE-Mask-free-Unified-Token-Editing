import torch
print("PyTorch version:", torch.__version__)
print("GPU:", torch.cuda.get_device_name())
x = torch.randn(1, device="cuda")
print("CUDA OK")
