import numpy as np
from skimage.metrics import structural_similarity as ssim
import pyiqa  # PyIQA supports multiple IQA metrics
import ssl
import torchvision.transforms as transforms
# painful to run the brisque model without updating the certificates
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import fft
from scipy.interpolate import interp1d


class ImageQualityMetrics:
    """
    A class to compute image quality metrics including MSE and PSNR, ....
    """

    @staticmethod
    def mse(image1, image2):
        """
        Compute the Mean Squared Error (MSE) between two images.

        Parameters:
        - image1 (numpy array or list): First image.
        - image2 (numpy array or list): Second image.

        Returns:
        - float: The computed MSE value.
        """
        img1 = np.array(image1)
        img2 = np.array(image2)
        return np.mean((img1 - img2) ** 2)

    @staticmethod
    def psnr(image1, image2):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

        Parameters:
        - image1 (numpy array or list): First image.
        - image2 (numpy array or list): Second image.

        Returns:
        - float: The computed PSNR value (higher is better).
        """
        mse_value = ImageQualityMetrics.mse(image1, image2)

        # If MSE is 0, images are identical
        if mse_value == 0:
            return float('inf')

        PIXEL_MAX = 255.0  # Maximum intensity for 8-bit images
        return 10 * np.log10((PIXEL_MAX ** 2) / mse_value)

    @staticmethod
    # Function to calculate SSIM
    def ssim(image1, image2):
        # Convert images to grayscale
        image1 = np.array(image1.convert('L'))  # Convert to grayscale
        image2 = np.array(image2.convert('L'))  # Convert to grayscale

        # Compute SSIM
        ssim_value, _ = ssim(image1, image2, full=True)
        return ssim_value

    @staticmethod
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

    @staticmethod
    def nrmse(original, compressed):
        """
        Compute Normalized Root Mean Square Error (NRMSE) between two images.

        Parameters:
        - original (numpy array): Ground truth image.
        - compressed (numpy array): Degraded or compressed image.

        Returns:
        - nrmse_value (float): The computed NRMSE value.
        """
        # Convert the images into numpy arryas
        original = np.array(original)
        compressed = np.array(compressed)
        # Ensure images have the same dimensions
        if original.shape != compressed.shape:
            raise ValueError("Error: Images must have the same dimensions.")

        # Compute Mean Squared Error (MSE)
        mse = np.mean((original - compressed) ** 2)

        # Compute Root Mean Square Error (RMSE)
        rmse = np.sqrt(mse)

        # Normalize by image intensity range (max - min)
        range_original = np.max(original) - np.min(original)

        # Avoid division by zero in case of constant images
        nrmse_value = rmse / range_original if range_original != 0 else float('inf')

        return nrmse_value

    @staticmethod
    # Function to calculate Signal-to-Noise Ratio (SNR)
    def snr(image, signal_mask):
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

    @staticmethod
    #Function to calculate brisque, nique, pique
    def NRImage(image):
        # Initialize PyIQA models
        brisque_model = pyiqa.create_metric('brisque')  # Blind/Referenceless IQA
        niqe_model = pyiqa.create_metric('niqe')  # Naturalness Image Quality Evaluator
        piqe_model = pyiqa.create_metric('piqe')  # Perception-based IQA

        # Convert to tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Compute NR-IQA scores
        brisque_score = brisque_model(image_tensor).item()
        niqe_score = niqe_model(image_tensor).item()
        piqe_score = piqe_model(image_tensor).item()

        return brisque_score, niqe_score, piqe_score

    @staticmethod
    # Function to calculate the blur score using the variance of laplacian
    def laplacian_blur(path, threshold=100):
        # upload the image
        # Load the image in grayscale
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Compute the Laplacian of the image
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Compute the variance (blur score)
        blur_score = np.var(laplacian)

        # Determine if the image is blurry based on the threshold
        is_blurry = blur_score < threshold

        return blur_score, is_blurry

    @staticmethod
    # Function to calculate the blur score using Fourier Transform
    def fft_blur(path, threshold=10):
        # upload the image
        # Load the image in grayscale
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

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

    @staticmethod
    # Function to calculate if the image is blur using wavelet
    def blur_wavelet(path, threshold=0.002):
        # Load image in grayscale
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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

    @staticmethod
    # Function to calculate the MTF
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
        roi = image[:, image.shape[1] // 2 - 5: image.shape[1] // 2 + 5]  # Take a wider region

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