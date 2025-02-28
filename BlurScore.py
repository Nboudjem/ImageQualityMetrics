import cv2
import numpy as np


# Function to calculate blur score using Variance of Laplacian
def detect_blur(image_path, threshold=100):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or invalid format")

    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance (blur score)
    blur_score = np.var(laplacian)

    # Determine if the image is blurry based on the threshold
    is_blurry = blur_score < threshold

    return blur_score, is_blurry


