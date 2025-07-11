import torch
import time
import argparse

def gb_to_elements(gb, dtype):
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    return int((gb * (1024**3)) / bytes_per_element)

def allocate_on_device(device, target_gb, dtype=torch.float32):
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    num_elements = gb_to_elements(target_gb, dtype)

    try:
        tensor = torch.empty(num_elements, dtype=dtype, device=f"cuda:{device}")
        print(f"[GPU {device}] Allocated {target_gb} GB successfully.")
        return tensor
    except RuntimeError as e:
        print(f"[GPU {device}] Allocation failed:", e)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupy GPU memory on selected devices.")
    parser.add_argument("--gb", type=float, required=True, help="Memory to occupy per GPU in GB.")
    parser.add_argument("--sleep", type=int, default=600, help="Hold time in seconds.")
    parser.add_argument("--devices", type=int, nargs="+", help="List of GPU device indices to use.")
    args = parser.parse_args()

    target_mem_gb = args.gb
    hold_time = args.sleep
    selected_devices = args.devices

    available_devices = torch.cuda.device_count()
    all_devices = list(range(available_devices))

    if selected_devices is None:
        selected_devices = all_devices
    else:
        invalid = [d for d in selected_devices if d not in all_devices]
        if invalid:
            raise ValueError(f"Invalid GPU indices: {invalid}. Available devices: {all_devices}")

    print(f"Using devices: {selected_devices}")

    tensors = []
    for device in selected_devices:
        tensor = allocate_on_device(device, target_mem_gb)
        if tensor is not None:
            tensors.append(tensor)

    print(f"Holding memory for {hold_time} seconds...")
    time.sleep(hold_time)

