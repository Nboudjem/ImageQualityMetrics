import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image


# Function to calculate SSIM
def calculate_ssim(image1, image2):
    # Convert images to grayscale
    image1 = np.array(image1.convert('L'))  # Convert to grayscale
    image2 = np.array(image2.convert('L'))  # Convert to grayscale

    # Compute SSIM
    ssim_value, _ = ssim(image1, image2, full=True)
    return ssim_value


# Load images
image1 = Image.open('image1.png')  # Replace with the path to your first image
image2 = Image.open('image2.png')  # Replace with the path to your second image

# Ensure both images are in the same size
if image1.size != image2.size:
    print("Images must have the same dimensions.")
else:
    # Calculate SSIM
    ssim_value = calculate_ssim(image1, image2)
    print(f"SSIM value: {ssim_value}")
