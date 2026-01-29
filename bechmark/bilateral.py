import time
import cv2
import torch
import fastcv
import numpy as np


def benchmark_bilateral(sizes=[1024, 2048, 4096, 8192], runs=10):
    results = []

    filter_size = 9
    sigma_color = 75.0
    sigma_space = 75.0

    for size in sizes:
        print(f"\n=== Benchmarking {size}x{size} image ===")

        img_np = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        img_torch = torch.from_numpy(img_np).cuda()

        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.bilateralFilter(img_np, filter_size, sigma_color, sigma_space)

        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000  # ms per run


        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.bilateral_filter(img_torch, filter_size, sigma_color, sigma_space)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000  # ms per run

        results.append((size, cv_time, fc_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUDA): {fc_time:.4f} ms")

    return results


if __name__ == "__main__":
    results = benchmark_bilateral()
    print("\n=== Final Results ===")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)")
    for size, cv_time, fc_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms")