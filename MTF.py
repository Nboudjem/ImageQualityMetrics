import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.interpolate import interp1d

def compute_mtf(image_path):
    """
    Compute the Modulation Transfer Function (MTF) and find MTF50.

    Parameters:
    - image_path (str): Path to the edge image.

    Returns:
    - frequencies (numpy array): Frequency axis.
    - mtf (numpy array): MTF values.
    - mtf50 (float): Frequency at which MTF drops to 50%.
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Error: Could not load the image.")

    # Select a region of interest (ROI) with the edge
    roi = image[:, image.shape[1] // 2 - 5 : image.shape[1] // 2 + 5]  # Take a wider region

    # Compute the Edge Spread Function (ESF)
    esf = np.mean(roi, axis=1)  # Ensure it's 1D

    # Check ESF validity before computing LSF
    if esf.ndim == 0 or esf.size == 0:
        raise ValueError("ESF computation failed: Check ROI selection or image contrast.")

    # Compute the Line Spread Function (LSF) by differentiating ESF
    lsf = np.diff(esf)

    # Compute the Modulation Transfer Function (MTF) using Fourier Transform
    mtf = np.abs(fft(lsf))
    mtf = mtf / np.max(mtf)  # Normalize to 1

    # Compute frequency axis
    frequencies = np.linspace(0, 1, len(mtf))

    # Find MTF50 (interpolating to find where MTF = 0.5)
    interp_func = interp1d(mtf, frequencies, kind="linear", bounds_error=False, fill_value="extrapolate")
    mtf50 = interp_func(0.5)  # Find the frequency where MTF = 0.5

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Edge Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.plot(esf, label="ESF")
    plt.title("Edge Spread Function (ESF)")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(frequencies, mtf, label="MTF")
    plt.axvline(mtf50, color='r', linestyle="--", label=f"MTF50 = {mtf50:.4f}")
    plt.title("Modulation Transfer Function (MTF)")
    plt.xlabel("Frequency (cycles/pixel)")
    plt.ylabel("MTF")
    plt.legend()

    plt.show()

    return frequencies, mtf, mtf50

# Example usage
image_path = "C:/Users/User/Desktop/ImageQuality/chest.jpg"  # Provide an edge image
frequencies, mtf, mtf50 = compute_mtf(image_path)

print(f"MTF50 Frequency: {mtf50:.4f} cycles/pixel")
