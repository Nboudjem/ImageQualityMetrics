import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_blur_wavelet(image_path, threshold=0.002):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or invalid format")

    # Perform 2D Discrete Wavelet Transform (DWT)
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2  # Approximation, Horizontal, Vertical, Diagonal

    # Compute energy of high-frequency components
    energy = np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)

    # Normalize energy by image size
    blur_score = energy / (image.shape[0] * image.shape[1])

    # Determine if image is blurry
    is_blurry = blur_score < threshold

    # Plot wavelet components
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(cH), cmap='gray')
    plt.title("Wavelet High-Frequency Detail")
    plt.axis("off")

    plt.show()

    return blur_score, is_blurry


# Example usage
image_path = "image.png"  # Replace with your image
blur_score, is_blurry = detect_blur_wavelet(image_path)

print(f"Wavelet Blur Score: {blur_score:.6f}")
print("Blurry Image: Yes" if is_blurry else "Blurry Image: No")

"""

Method	                        Strengths	                        Limitations
Laplacian Variance	            Fast and simple	                    Sensitive to noise
FFT-Based Blur Detection	    Robust for detecting motion blur	Sensitive to image size variations
Wavelet-Based Blur Detection	Works well for natural images	    Computationally expensive
"""