import torch
import time

def squeeze_gpu_memory(min_remaining_gb=5):
    if not torch.cuda.is_available():
        print("未检测到 CUDA 环境")
        return

    device_count = torch.cuda.device_count()
    print(f"检测到 {device_count} 块显卡。开始挤压显存...")

    # 存储所有申请到的 Tensor，防止被垃圾回收
    holders = [[] for _ in range(device_count)]
    
    # 转换单位
    MIN_REMAINING = min_remaining_gb * 1024 * 1024 * 1024  # 5GB
    
    # 初始申请步长（从 2GB 开始尝试，逐步减小到 128MB）
    STEP_SIZES = [2048, 1024, 512, 256, 128] # 单位：MB

    for i in range(device_count):
        torch.cuda.set_device(i)
        device_name = torch.cuda.get_device_name(i)
        print(f"\n--- 正在处理显卡 {i}: {device_name} ---")

        for step_mb in STEP_SIZES:
            step_bytes = step_mb * 1024 * 1024
            count = 0
            
            while True:
                # 检查当前剩余显存
                total_mem = torch.cuda.get_device_properties(i).total_memory
                reserved_mem = torch.cuda.memory_reserved(i)
                allocated_mem = torch.cuda.memory_allocated(i)
                
                # 估算系统剩余可用显存 (Total - Reserved)
                # 注意：nvidia-smi 显示的 free 可能更准，但 torch 内部只能感知自己申请的
                free_mem = total_mem - reserved_mem - allocated_mem
                
                # 如果剩余空间已经小于目标阈值，停止当前卡的申请
                if free_mem < MIN_REMAINING:
                    break
                
                try:
                    # 申请一块显存
                    # float32 占用 4 字节，所以元素个数 = 字节数 // 4
                    shape = (step_bytes // 4,)
                    t = torch.empty(shape, dtype=torch.float32, device=f'cuda:{i}')
                    holders[i].append(t)
                    count += 1
                except RuntimeError:
                    # 如果申请失败（Out of Memory），说明当前 step 太大，跳出循环尝试更小的 step
                    break
            
            if count > 0:
                print(f"以 {step_mb}MB 步长成功申请了 {count} 块显存")

        final_free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3
        print(f"显卡 {i} 处理完毕，预计剩余可用显存: {final_free:.2f} GB")

    print("\n所有显卡已达到目标占用状态。按 Ctrl+C 释放显存并退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在释放显存...")
        del holders
        torch.cuda.empty_cache()
        print("显存已释放。")

if __name__ == "__main__":
    # 你可以修改此处的参数，例如改为 2 代表保留 2GB
    squeeze_gpu_memory(min_remaining_gb=5)
