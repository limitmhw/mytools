import torch

try:
    import cutlass
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

torch.manual_seed(42)

M, K, N = 10240, 10240, 102400
loops = 50

device = "cuda"

A_int8 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
B_int8 = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)

A_fp16 = A_int8.to(torch.float16)
B_fp16 = B_int8.to(torch.float16)

linear_layer = torch.nn.Linear(K, N, bias=False).to(device, dtype=torch.float16)
with torch.no_grad():
    linear_layer.weight.copy_(B_fp16.T)

# torch._int_mm requires column-major B (contiguous transposed)
B_int8_for_intmm = B_int8.t().contiguous().t()

if CUTLASS_AVAILABLE:
    C_int32 = torch.zeros(M, N, dtype=torch.int32, device=device)
    D_int32 = torch.zeros(M, N, dtype=torch.int32, device=device)

    plan = cutlass.op.Gemm(
        element_A=cutlass.DataType.s8,
        element_B=cutlass.DataType.s8,
        element_C=cutlass.DataType.s32,
        element_D=cutlass.DataType.s32,
        layout=cutlass.LayoutType.RowMajor
    )

    # Trigger CUTLASS JIT compilation
    plan.run(A_int8, B_int8, C_int32, D_int32, alpha=1, beta=0)
    torch.cuda.synchronize()

# Warmup
for _ in range(10):
    torch.nn.functional.linear(A_fp16, linear_layer.weight)
    torch._int_mm(A_int8, B_int8_for_intmm)
    if CUTLASS_AVAILABLE:
        plan.run(A_int8, B_int8, C_int32, D_int32, alpha=1, beta=0)
torch.cuda.synchronize()

# --- Benchmark 1: torch.nn.Linear (FP16 cuBLAS) ---
fp16_start = torch.cuda.Event(enable_timing=True)
fp16_end = torch.cuda.Event(enable_timing=True)
fp16_start.record()
for _ in range(loops):
    out_fp16 = torch.nn.functional.linear(A_fp16, linear_layer.weight)
fp16_end.record()
torch.cuda.synchronize()
linear_time_ms = fp16_start.elapsed_time(fp16_end)

# --- Benchmark 2: CUTLASS INT8 GEMM (Python wrapper) ---
cutlass_time_ms = None
if CUTLASS_AVAILABLE:
    cutlass_start = torch.cuda.Event(enable_timing=True)
    cutlass_end = torch.cuda.Event(enable_timing=True)
    cutlass_start.record()
    for _ in range(loops):
        plan.run(A_int8, B_int8, C_int32, D_int32, alpha=1, beta=0)
    cutlass_end.record()
    torch.cuda.synchronize()
    cutlass_time_ms = cutlass_start.elapsed_time(cutlass_end)

# --- Benchmark 3: torch._int_mm (INT8 cuBLAS, fair comparison) ---
intmm_start = torch.cuda.Event(enable_timing=True)
intmm_end = torch.cuda.Event(enable_timing=True)
intmm_start.record()
for _ in range(loops):
    out_intmm = torch._int_mm(A_int8, B_int8_for_intmm)
intmm_end.record()
torch.cuda.synchronize()
intmm_time_ms = intmm_start.elapsed_time(intmm_end)

print(f"Matrix size: M={M}, K={K}, N={N}, loops={loops}")
print(f"{'Method':<40} {'Avg time (ms)':>14} {'vs FP16':>10}")
print("-" * 66)
print(f"{'torch.nn.Linear (FP16 cuBLAS)':<40} {linear_time_ms/loops:>14.3f} {'1.00x':>10}")
if cutlass_time_ms is not None:
    print(f"{'CUTLASS INT8 (Python wrapper)':<40} {cutlass_time_ms/loops:>14.3f} {linear_time_ms/cutlass_time_ms:>9.2f}x")
else:
    print(f"{'CUTLASS INT8 (Python wrapper)':<40} {'SKIPPED (cutlass not installed)':>26}")
print(f"{'torch._int_mm (INT8 cuBLAS)':<40} {intmm_time_ms/loops:>14.3f} {linear_time_ms/intmm_time_ms:>9.2f}x")
