import torch

def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA built? {torch.backends.cuda.is_built()}")
    
    if torch.cuda.is_available():
        print("CUDA is available. GPU detected:")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. No GPU detected.")
        print("Possible reasons:")
        print("1. No NVIDIA GPU present")
        print("2. NVIDIA drivers not installed")
        print("3. PyTorch installed without CUDA support")
        print("4. You are poor")

if __name__ == "__main__":
    check_cuda()