import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
    x = torch.randn(1, device="cuda")
    print("Tensor on CUDA:", x)
    print("Success!")
