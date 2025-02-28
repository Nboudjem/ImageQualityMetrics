import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_blur_fft(image_path, threshold=10):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or invalid format")

    # Compute the 2D FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(fft_shift)

    # Compute mean frequency value as blur indicator
    blur_score = np.mean(magnitude_spectrum)

    # Determine if the image is blurry
    is_blurry = blur_score < threshold

    # Plot magnitude spectrum
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    plt.title("FFT Magnitude Spectrum")
    plt.axis("off")

    plt.show()

    return blur_score, is_blurry


# Example usage
image_path = "image.png"  # Replace with your image
blur_score, is_blurry = detect_blur_fft(image_path)

print(f"FFT Blur Score: {blur_score:.2f}")
print("Blurry Image: Yes" if is_blurry else "Blurry Image: No")
