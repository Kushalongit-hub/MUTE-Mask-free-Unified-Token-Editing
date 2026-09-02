import torch
import time

print("Starting long CUDA computation test...")
torch.cuda.reset_peak_memory_stats()

# Simulate a long computation
a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
print("Allocated tensor, starting matmul...")

for i in range(20):
    start = time.time()
    b = torch.mm(a, a.T)
    c = torch.mm(b, a)
    torch.cuda.synchronize()
    print(f"Iteration {i}: {time.time() - start:.2f}s, peak mem: {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB")
    del b, c
    torch.cuda.empty_cache()

print("Long computation test passed!")
