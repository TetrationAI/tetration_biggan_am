import torch
import time


# Results for comparison 
"""
Using device: cuda (NVIDIA RTX 4090)
Average time per multiplication: 0.000005 seconds.

Using device: cpu
Average time per multiplication: Many seconds... more than 10
"""

def test_gpu_performance(size=1024, iterations=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create random matrices
    matrix1 = torch.rand(size, size, device=device)
    matrix2 = torch.rand(size, size, device=device)

    # Warm up
    for _ in range(10):
        _ = torch.mm(matrix1, matrix2)

    # Timing the matrix multiplication
    start_time = time.time()
    for _ in range(iterations):
        _ = torch.mm(matrix1, matrix2)
    end_time = time.time()

    print(f"Average time per multiplication: {(end_time - start_time) / iterations:.6f} seconds")

# Test with a 1024x1024 matrix, repeated 10 times
test_gpu_performance(2048, 1000)
#gpu:  Average time per multiplication: 0.002661 seconds
#cpu:  Average time per multiplication: 0.072930