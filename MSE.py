import numpy as np
from PIL import Image


# Function to calculate Mean Squared Error (MSE)
def mse(image1, image2):
    # Convert images to numpy arrays
    img1 = np.array(image1)
    img2 = np.array(image2)

    # Compute MSE (mean squared error)
    error = np.sum((img1 - img2) ** 2)
    mse_value = error / float(img1.size)
    return mse_value


# Load images
image1 = Image.open('image1.png')  # Replace with the path to your first image
image2 = Image.open('image2.png')  # Replace with the path to your second image

# Ensure both images have the same size
if image1.size != image2.size:
    print("Images must have the same dimensions.")
else:
    # Calculate MSE
    mse_value = mse(image1, image2)
    print(f"MSE value: {mse_value}")
