import numpy as np
from PIL import Image
import ImageQualityMetricsCalss as metrics

'''
Main file that execute the different functions of the class ImageQualityMetricsClass
'''

# Load images
img1 = Image.open('C:/Users/User/Desktop/ImageQuality/chest.jpg')  # Replace with the path to your image
img2 = Image.open('C:/Users/User/Desktop/ImageQuality/chest.jpg')  # Replace with the path to your image

# Ensure images are loaded correctly
if img1 is None or img2 is None:
    raise ValueError("Error: One or both images could not be loaded.")

path = 'C:/Users/User/Desktop/ImageQuality/chest.jpg'

# Ensure both images are in the same mode and size
if img1.size != img2.size:
    print("Images must have the same dimensions.")
else:
    # Calculate MSE
    mse_value = metrics.ImageQualityMetrics.mse(img1, img2)
    print(f"MSE: {mse_value:.4f}")

    #Calculate PSNR
    psnr_value = metrics.ImageQualityMetrics.psnr(img1, img2)
    print(f"PSNR: {psnr_value:.4f} dB")

    #Calculate SSIM
    ssim_value = metrics.ImageQualityMetrics.ssim(img1, img2)
    print(f"SSIM value: {ssim_value}")

    #Calculate CNR
    # Create a simple mask for signal (foreground) and background regions
    # For example, assume the foreground is a region of interest in the center of the image
    # TODO: code to be fixed here! the shape should be 2D instead of 3D
    height, width, _ = np.array(img1).shape
    signal_mask = np.zeros((height, width), dtype=bool)
    background_mask = np.ones((height, width), dtype=bool)
    # Define signal region (e.g., center of the image)
    signal_mask[100:200, 100:200] = True  # Example: foreground in a 100x100 region in the center
    # Define background region (e.g., remaining part of the image)
    background_mask[:100, :] = False  # Example: top part is background
    background_mask[200:, :] = False  # Example: bottom part is background
    background_mask[:, :100] = False  # Example: left part is background
    background_mask[:, 200:] = False  # Example: right part is background
    cnr_value = metrics.ImageQualityMetrics.cnr(img1, signal_mask, background_mask)
    print(f"CNR value: {cnr_value}")

    #Calculate NRMSE
    nrmse_value = metrics.ImageQualityMetrics.nrmse(img1, img2)
    print(f"NRMSE Score: {nrmse_value:.4f}")

    # Calculate SNR
    image = img1.convert('L')  # Convert to grayscale
    # Define a simple mask (Example: Center region as signal)
    height, width= np.array(image).shape
    signal_mask = np.zeros((height, width), dtype=bool)
    signal_mask[100:200, 100:200] = True  # Example: A 100x100 region in the center
    snr_value = metrics.ImageQualityMetrics.snr(image, signal_mask)
    print(f"SNR value: {snr_value} dB")

    # Calculate NoReference Image Quality
    brisque_score, niqe_score, piqe_score = metrics.ImageQualityMetrics.NRImage(image)
    '''
    Common No-Reference Image Quality Metrics:
    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    NIQE (Naturalness Image Quality Evaluator)
    PIQE (Perception-based Image Quality Evaluator)
    '''
    # Print results
    print(f"BRISQUE Score: {brisque_score:.2f} (Lower is better)")
    print(f"NIQE Score: {niqe_score:.2f} (Lower is better)")
    print(f"PIQE Score: {piqe_score:.2f} (Lower is better)")

    # Calculate the blur score using Laplacian
    blur_score, is_blury  = metrics.ImageQualityMetrics.laplacian_blur(path)
    print(f"Blur score: {blur_score: .2f} (Higher is better)")
    print("Blurry Image: Yes" if is_blury else "Blurry Image: No according to given threshold")

    # Calculate the blur score using Fourier Transform
    fft_blur_score, fft_is_blury  = metrics.ImageQualityMetrics.fft_blur(path)
    print(f"FFt Blur score: {fft_blur_score: .2f} (Higher is better)")
    print("Blurry Image: Yes" if fft_is_blury else "Blurry Image: No accoriding to threshold given")

    # Calculate the blur score using wavelet
    wavelet_blur_score, wavelet_is_blury  = metrics.ImageQualityMetrics.blur_wavelet(path)
    print(f"Wavelet Blur score: {wavelet_blur_score: .2f} (Higher is better)")
    print("Blurry Image: Yes" if wavelet_is_blury else "Blurry Image: No according to threshold given")

    # Calculate the MTF
    frequencies, mtf, mtf50 = metrics.ImageQualityMetrics.compute_mtf(path)
    print(f"MTF50 Frequency: {mtf50:.4f} cycles/pixel")
    '''
    Example: Interpreting an MTF Plot
    *Good Imaging System (Sharp)
    
    MTF starts at 1.0 at low frequencies.
    Gradually decreases.
    MTF50 at a high frequency (good sharpness).
    Still some contrast at high frequencies (good resolution).
    
    *Poor Imaging System (Blurry)
    
    MTF drops sharply.
    MTF50 at a low frequency (early loss of contrast).
    High frequencies show almost no contrast.
    '''





