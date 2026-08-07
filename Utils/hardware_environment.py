# Utils/hardware_environment.py

import platform

import torch


def print_pytorch_cuda_summary(
    cuda_cores_per_sm: int | None = 128,
) -> None:
    """
    Print Python, PyTorch, CUDA, and hardware information.

    The function automatically uses PyTorch's current CUDA device when CUDA
    is available. If CUDA is unavailable, it reports a CPU fallback instead.

    Parameters
    ----------
    cuda_cores_per_sm:
        CUDA cores per streaming multiprocessor used to derive the total
        CUDA-core count.

        Set to None to omit the derived CUDA-core estimate.

        This value is externally supplied because PyTorch reports the number
        of streaming multiprocessors but not the number of CUDA cores per SM.
    """

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()

    print("Python and PyTorch")
    print("-" * 56)
    print(f"Python version:              {platform.python_version()}")
    print("PyTorch import:              Successful")
    print(f"PyTorch version:             {torch.__version__}")
    print(f"PyTorch CUDA runtime:        {torch.version.cuda}")
    print(f"CUDA available:              {cuda_available}")
    print(f"CUDA GPU count:              {gpu_count}")

    if not cuda_available:
        print()
        print("CPU fallback")
        print("-" * 56)
        print("Selected device:             CPU")
        print("Reason:                      No CUDA-accessible GPU")
        return

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)

    print()
    print("Values reported by CUDA/PyTorch")
    print("-" * 56)
    print(f"CUDA device index:           {device}")
    print(f"GPU model:                   {properties.name}")
    print(f"Compute capability:          {major}.{minor}")
    print(f"CUDA architecture:           sm_{major}{minor}")
    print(
        f"Total VRAM:                  "
        f"{properties.total_memory / 1024**3:.2f} GiB"
    )
    print(
        f"Streaming multiprocessors:   "
        f"{properties.multi_processor_count}"
    )
    print(
        f"Maximum threads per block:   "
        f"{properties.max_threads_per_block}"
    )
    print(f"Warp size:                   {properties.warp_size}")

    if cuda_cores_per_sm is not None:
        cuda_core_count = (
            properties.multi_processor_count
            * cuda_cores_per_sm
        )

        print()
        print("Externally derived value")
        print("-" * 56)
        print(f"Assumed CUDA cores per SM:   {cuda_cores_per_sm}")
        print(f"Derived CUDA-core count:     {cuda_core_count}")
        print(
            "Derivation:                  "
            f"{properties.multi_processor_count} SMs × "
            f"{cuda_cores_per_sm} cores/SM"
        )