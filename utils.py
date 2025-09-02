import torch
import logging

def set_seeds(seed: int=42):
    """Sets the seed for generating random numbers.

    Args:
        seed (int, optional): seed value. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

# set device agnostic code
def device_setup() -> torch.device:
    """
    Set up device-agnostic configuration 

    Returns:
        torch.device: cuda or cpu
    """
    # check if gpu is available or not
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if device is cuda, then clear the cache first
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Using device: {device}")

    return device

# Set training timer

def training_time(start: float,
                  end: float,
                  device: torch.device='cpu') -> None:
    """Prints total time taken for training

    Args:
        start (float): time at which training started
        end (float): tiem at which training finished
        device (torch.device): device on which training happened
    """
    total_time = end - start
    print(f"Training time on {device}: {total_time:.3f} seconds")

# # 
# # Creata a logging object
# #

# logging.basicConfig(
#     format="[ %(asctime)s - %(levelname)8s - %(filename)s:%(lineno)d ]"
#            "%(message)s",
#     datefmt="%d-%b-%y  %H:%M:%S",
#     level=logging.INFO
# )