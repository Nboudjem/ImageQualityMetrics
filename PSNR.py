import numpy as np
from PIL import Image

'''
Result: The PSNR value will be printed in decibels (dB). A higher PSNR indicates that the images are more similar, and the quality is better.
'''


# Function to calculate Mean Squared Error (MSE)
def mse(image1, image2):
    # Convert images to numpy arrays
    img1 = np.array(image1)
    img2 = np.array(image2)

    # Compute MSE
    return np.mean((img1 - img2) ** 2)


# Function to calculate PSNR
def psnr(image1, image2):
    # Compute MSE
    mse_value = mse(image1, image2)

    # If MSE is 0, the images are identical
    if mse_value == 0:
        return float('inf')

    # Compute PSNR
    PIXEL_MAX = 255.0  # for 8-bit images
    psnr_value = 10 * np.log10((PIXEL_MAX ** 2) / mse_value)
    return psnr_value


# Load images
image1 = Image.open('C:/Users/User/Desktop/ImageQuality/chest.jpg')  # Replace with the path to your image
image2 = Image.open('C:/Users/User/Desktop/ImageQuality/chest.jpg')  # Replace with the path to your image

# Ensure both images are in the same mode and size
if image1.size != image2.size:
    print("Images must have the same dimensions.")
else:
    # Calculate PSNR
    psnr_value = psnr(image1, image2)
    print(f"PSNR value: {psnr_value} dB")
