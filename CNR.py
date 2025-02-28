import numpy as np
from PIL import Image


# Function to calculate Contrast-to-Noise Ratio (CNR)
def cnr(image, signal_mask, background_mask):
    # Convert image to numpy array
    img = np.array(image)

    # Signal region (foreground) - mean and standard deviation
    signal_region = img[signal_mask]
    mu_s = np.mean(signal_region)  # mean of the signal region
    sigma_s = np.std(signal_region)  # standard deviation of the signal region

    # Background region - mean and standard deviation
    background_region = img[background_mask]
    mu_b = np.mean(background_region)  # mean of the background region
    sigma_b = np.std(background_region)  # standard deviation of the background region

    # Compute Contrast-to-Noise Ratio
    cnr_value = np.abs(mu_s - mu_b) / np.sqrt(sigma_s ** 2 + sigma_b ** 2)
    return cnr_value


# Load the image
image = Image.open('image.png')  # Replace with the path to your image

# Create a simple mask for signal (foreground) and background regions
# For example, assume the foreground is a region of interest in the center of the image
height, width = np.array(image).shape
signal_mask = np.zeros((height, width), dtype=bool)
background_mask = np.ones((height, width), dtype=bool)

# Define signal region (e.g., center of the image)
signal_mask[100:200, 100:200] = True  # Example: foreground in a 100x100 region in the center

# Define background region (e.g., remaining part of the image)
background_mask[:100, :] = False  # Example: top part is background
background_mask[200:, :] = False  # Example: bottom part is background
background_mask[:, :100] = False  # Example: left part is background
background_mask[:, 200:] = False  # Example: right part is background

# Calculate CNR
cnr_value = cnr(image, signal_mask, background_mask)
print(f"CNR value: {cnr_value}")
