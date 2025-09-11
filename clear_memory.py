import torch
import time

def allocate_memory(device_id=0):
    torch.cuda.set_device(device_id)
    allocated_tensors = []  # 保存申请的显存，防止被GC释放

    # 从大到小逐步尝试申请的大小（单位：字节）
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
                # 每个 float32=4B，所以要除以4
                num_elements = size // 4
                tensor = torch.empty(num_elements, dtype=torch.float32, device=f"cuda:{device_id}")
                allocated_tensors.append(tensor)
                print(f"[+] Allocated {size / 1024**2:.2f} MB on cuda:{device_id}, total tensors: {len(allocated_tensors)}")
                success = True
                break
            except RuntimeError:
                continue  # 当前大小分配失败，尝试更小的

        if not success:
            print("[-] No more memory available, sleeping 500ms...")
            time.sleep(0.5)


if __name__ == "__main__":
    allocate_memory(device_id=7)  # 默认申请 GPU0，可改成其他卡
