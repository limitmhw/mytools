import torch
import time
import numpy as np
import threading

def allocate_gpu_memory(device_id=0):
    torch.cuda.set_device(device_id)
    allocated_tensors = []

    sizes = [
        100 * 1024**3,   # 100 GB
        10 * 1024**3,    # 10 GB
        1 * 1024**3,     # 1 GB
        100 * 1024**2,   # 100 MB
        10 * 1024**2,    # 10 MB
        1 * 1024**2,     # 1 MB
        100 * 1024,      # 100 KB
        10 * 1024,       # 10 KB
        1024,            # 1 KB
        100,             # 0.1 KB
    ]

    while True:
        success = False
        for size in sizes:
            try:
                num_elements = size // 4  # float32 占 4 字节
                tensor = torch.empty(num_elements, dtype=torch.float32, device=f"cuda:{device_id}")
                allocated_tensors.append(tensor)
                print(f"[GPU] Allocated {size / 1024**2:.2f} MB on cuda:{device_id}, total blocks: {len(allocated_tensors)}")
                success = True
                break
            except RuntimeError:
                continue

        if not success:
            print("[GPU] No more memory available, sleeping 500ms...")
            time.sleep(0.5)


def allocate_cpu_memory():
    allocated_arrays = []

    sizes = [
        10 * 1024**3,   # 10 GB
        1 * 1024**3,    # 1 GB
        100 * 1024**2,  # 100 MB
        10 * 1024**2,   # 10 MB
        1 * 1024**2,    # 1 MB
        100 * 1024,     # 100 KB
        10 * 1024,      # 10 KB
        1024,           # 1 KB
        100,            # 0.1 KB
    ]

    while True:
        success = False
        for size in sizes:
            try:
                arr = np.empty(size // 8, dtype=np.float64)
                arr.fill(1)  # ⚠️ 强制写入，确保物理内存真正占用
                allocated_arrays.append(arr)
                print(f"[CPU-MEM] Allocated {size / 1024**2:.2f} MB RAM, total blocks: {len(allocated_arrays)}")
                success = True
                break
            except MemoryError:
                continue

        if not success:
            print("[CPU-MEM] No more memory available, sleeping 500ms...")
            time.sleep(0.5)


def burn_cpu_compute():
    print("[CPU-COMPUTE] Starting busy loop...")
    x, y = 1, 2
    while True:
        x, y = y, x + y  # 死循环，持续吃 CPU


if __name__ == "__main__":
    # 启动多 GPU 占用
    for i in range(torch.cuda.device_count()):
        threading.Thread(target=allocate_gpu_memory, args=(i,), daemon=True).start()

    # 启动 CPU 内存占用
    threading.Thread(target=allocate_cpu_memory, daemon=True).start()

    # 启动 CPU 算力占用
    burn_cpu_compute()
