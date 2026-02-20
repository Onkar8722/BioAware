import torch

def check_pytorch_cuda():
    print("--- PyTorch & CUDA Diagnostics ---")
    
    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check the CUDA version PyTorch was compiled with (should say 12.1)
    print(f"PyTorch built with CUDA Version: {torch.version.cuda}")
    
    # Check if CUDA is available and accessible
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available to PyTorch? {cuda_available}")
    
    if cuda_available:
        # Get the number of available GPUs
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        
        # Get the name of the current GPU
        current_device = torch.cuda.current_device()
        print(f"Active GPU Name: {torch.cuda.get_device_name(current_device)}")
        
        # Do a quick tensor test on the GPU
        x = torch.rand(3, 3).cuda()
        print("\nSuccess! Created a test tensor directly on the GPU:")
        print(x)
    else:
        print("\nUh oh. PyTorch cannot see the GPU. We might need to check your environment paths or drivers.")

if __name__ == "__main__":
    check_pytorch_cuda()