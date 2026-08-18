import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "NONE")
x = torch.randn(1000, 1000, device="cuda")
y = (x @ x).sum().item()
print("GPU matmul OK, result is finite:", y == y)
