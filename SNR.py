import numpy as np
from PIL import Image


# Function to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(image, signal_mask):
    # Convert image to numpy array
    img = np.array(image)

    # Extract signal region using the mask
    signal_region = img[signal_mask]

    # Calculate mean and standard deviation
    mu_s = np.mean(signal_region)  # Mean intensity of signal
    sigma_s = np.std(signal_region)  # Standard deviation (assumed noise)

    # Compute SNR
    snr_value = mu_s / sigma_s if sigma_s != 0 else float('inf')
    return snr_value


# Load the image
image = Image.open('image.png').convert('L')  # Convert to grayscale

# Define a simple mask (Example: Center region as signal)
height, width = np.array(image).shape
signal_mask = np.zeros((height, width), dtype=bool)
signal_mask[100:200, 100:200] = True  # Example: A 100x100 region in the center

# Calculate SNR
snr_value = calculate_snr(image, signal_mask)
print(f"SNR value: {snr_value} dB")
