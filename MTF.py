import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, fftshift


# Function to calculate MTF from an image
def calculate_mtf(image, edge_line=None):
    if edge_line is None:
        # Find the middle row of the image for a simple profile
        edge_line = image[image.shape[0] // 2, :]

    # Smooth the edge profile (apply Gaussian filter to reduce noise)
    edge_line_smoothed = gaussian_filter(edge_line, sigma=2)

    # Line Spread Function (LSF) is the derivative of the Edge Spread Function (ESF)
    lsf = np.gradient(edge_line_smoothed)

    # Calculate the Fourier Transform of the Line Spread Function (LSF)
    fft_lsf = fft(lsf)

    # Compute the Modulation Transfer Function (MTF)
    mtf = np.abs(fftshift(fft_lsf))

    # Create the frequency axis (in cycles per pixel)
    freqs = np.fft.fftfreq(len(mtf))

    # Plot MTF curve
    plt.plot(np.abs(freqs), mtf, label="MTF")
    plt.title('Modulation Transfer Function (MTF)')
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('MTF')
    plt.grid(True)
    plt.legend()
    plt.show()

    return mtf, freqs


# Load a sample image (e.g., an edge or a line pattern)
image = color.rgb2gray(data.camera())  # Grayscale image (camera image)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# Call the function to calculate MTF
mtf, freqs = calculate_mtf(image)
