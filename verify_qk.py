import numpy as np
import torch

seq_len = 512
head_dim = 64

Q = np.fromfile("Q.bin", dtype=np.float32).reshape(seq_len, head_dim)
K = np.fromfile("K.bin", dtype=np.float32).reshape(seq_len, head_dim)
S_cuda = np.fromfile("S.bin", dtype=np.float32).reshape(seq_len, seq_len)

Q_t = torch.from_numpy(Q)
K_t = torch.from_numpy(K)
S_torch = (Q_t @ K_t.T).numpy()

max_diff = np.max(np.abs(S_cuda - S_torch))
mean_diff = np.mean(np.abs(S_cuda - S_torch))

print(f"Max absolute difference: {max_diff}")
print(f"Mean absolute difference: {mean_diff}")
print(f"S_cuda[0][0] = {S_cuda[0][0]}, S_torch[0][0] = {S_torch[0][0]}")
print(f"S_cuda[1][0] = {S_cuda[1][0]}, S_torch[1][0] = {S_torch[1][0]}")

if max_diff < 1e-3:
    print("PASS: CUDA kernel matches PyTorch")
else:
    print("FAIL: mismatch detected")
