import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=2, help="Size of square matrix to multiply")
args = parser.parse_args()

if not torch.cuda.is_available():
    print("CUDA is not available, using CPU.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:1")
    print(f"Using CUDA device: {torch.cuda.get_device_name(1)}")

size1 = args.size

A = torch.randn(size1, size1, device=device)
print("init A success")
B = torch.randn(size1, size1, device=device)
print("init B success")

start_time = time.time()
C = torch.matmul(A, B)
end_time = time.time()

print(f"Matrix multiplication of size {size1}x{size1} on {device} took: {end_time - start_time:.4f} seconds")

# 可以选择性验证结果的正确性 (例如与 CPU 计算结果对比，或简单检查一些元素)
# print(C[:5,:5])  # 打印结果矩阵的一部分

