import cv2
import numpy as np


def calculate_nrmse(original, compressed):
    """
    Compute Normalized Root Mean Square Error (NRMSE) between two images.

    Parameters:
    - original (numpy array): Ground truth image.
    - compressed (numpy array): Degraded or compressed image.

    Returns:
    - nrmse_value (float): The computed NRMSE value.
    """
    # Ensure images have the same dimensions
    if original.shape != compressed.shape:
        raise ValueError("Error: Images must have the same dimensions.")

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original - compressed) ** 2)

    # Compute Root Mean Square Error (RMSE)
    rmse = np.sqrt(mse)

    # Normalize by image intensity range (max - min)
    range_original = np.max(original) - np.min(original)

    # Avoid division by zero in case of constant images
    nrmse_value = rmse / range_original if range_original != 0 else float('inf')

    return nrmse_value


# Load two images (convert to grayscale for comparison)
original = cv2.imread("original.png", cv2.IMREAD_GRAYSCALE)
compressed = cv2.imread("compressed.png", cv2.IMREAD_GRAYSCALE)

# Ensure images are loaded correctly
if original is None or compressed is None:
    raise ValueError("Error: One or both images could not be loaded.")

# Compute NRMSE
nrmse_value = calculate_nrmse(original, compressed)

print(f"NRMSE Score: {nrmse_value:.4f}")
