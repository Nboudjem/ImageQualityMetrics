# **Image Quality Metrics in Python**

This repository provides implementations of several image quality metrics used in image processing and computer vision. These metrics are crucial for evaluating the performance of imaging systems and assessing image quality objectively. They are commonly used in areas like medical imaging, image compression, denoising, and photography.

The metrics are written separately in different Python files, but are also gathered as a class in the file ImageQualityClass.py, which can be run under the file Main.py.

Let me know if you need further adjustments!

## **Image Quality Metrics Implemented**

The following image quality metrics are implemented in this repository:

### 1. **Peak Signal-to-Noise Ratio (PSNR)**
   - **Description:** PSNR is a widely used metric for measuring the quality of reconstructed images compared to the original image. It is expressed in decibels (dB) and measures the ratio between the maximum possible signal power and the noise introduced in the image.
   - **Formula:**  
     \[
     \text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)
     \]
     where `MAX` is the maximum pixel value of the image and `MSE` is the mean squared error between the original and distorted images.
   - **Usage:** High PSNR values indicate better image quality.

### 2. **Structural Similarity Index (SSIM)**
   - **Description:** SSIM is a perceptual metric that measures the structural similarity between two images. It takes into account luminance, contrast, and structure to provide a more accurate measure of perceived image quality than traditional metrics like PSNR.
   - **Formula:**  
     \[
     \text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
     \]
     where \(\mu_x\) and \(\mu_y\) are the mean pixel values, \(\sigma_x\) and \(\sigma_y\) are the standard deviations, and \(\sigma_{xy}\) is the covariance of the two images.
   - **Usage:** SSIM ranges from -1 to 1, where 1 means the images are identical.

### 3. **Mean Squared Error (MSE)**
   - **Description:** MSE is a simple metric that computes the average squared difference between the original and distorted images. A lower MSE indicates better quality.
   - **Formula:**  
     \[
     \text{MSE}(x, y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2
     \]
   - **Usage:** MSE is sensitive to pixel-wise differences, but does not capture perceptual aspects like SSIM.

### 4. **Contrast-to-Noise Ratio (CNR)**
   - **Description:** CNR measures the contrast between the signal (object) and background noise. It is used to assess the visibility of objects in noisy images.
   - **Formula:**  
     \[
     \text{CNR} = \frac{|\mu_s - \mu_b|}{\sigma_b}
     \]
     where \(\mu_s\) and \(\mu_b\) are the mean values of the object and background regions, and \(\sigma_b\) is the standard deviation of the background.
   - **Usage:** Higher CNR values indicate better contrast and visibility of objects in the image.

### 5. **Modulation Transfer Function (MTF)**
   - **Description:** MTF quantifies how well the imaging system preserves the contrast of spatial details. It is derived from the Edge Spread Function (ESF) and describes the system’s ability to resolve fine details.
   - **Usage:** A higher MTF value indicates better sharpness, with MTF50 (the frequency at which MTF drops to 50%) being a key indicator of image resolution.

### 6. **Signal-to-Noise Ratio (SNR)**
   - **Description:** SNR measures the ratio of signal strength to noise in an image. It is used to determine the quality of an image in terms of its noise levels.
   - **Formula:**  
     \[
     \text{SNR} = \frac{\mu}{\sigma}
     \]
     where \(\mu\) is the mean value of the signal and \(\sigma\) is the standard deviation of the noise.
   - **Usage:** A higher SNR indicates a cleaner image with less noise.

### 7. **No-Reference Image Quality Metrics**
   - **Description:** These metrics do not require a reference image and are used to evaluate image quality based on certain features (e.g., sharpness, noise, blur) without comparison to an original image.
   - **Usage:** Useful in scenarios where the reference image is not available (e.g., image enhancement or real-time video processing).

### 8. **Blur Detection**
   - **Description:** This metric evaluates the sharpness or blurriness of an image. It typically uses edge detection methods or gradient-based techniques to assess the clarity of image features.
   - **Usage:** Blur detection is commonly used in image preprocessing for sharpening or in determining the quality of a captured image.

### 9. **Normalized Root Mean Square Error (NRMSE)**
   - **Description:** NRMSE is a normalized version of the RMSE (Root Mean Squared Error) that accounts for the scale of the image. It gives a relative error between the original and the distorted image.
   - **Usage:** NRMSE is used to assess the error in pixel values while normalizing for image intensity.

## **Usage**

To use these metrics, simply pass two images (or an image and its reference) to the respective functions. The metrics will return a score indicating the quality of the image.

## **Dependencies**
numpy
opencv-python
scikit-image
matplotlib
scipy

## **Contributing** 
Feel free to contribute to this repository by submitting pull requests. You can add more image quality metrics or improve the current implementations.

### Example:
```python
import cv2
from image_quality_metrics import psnr, ssim, mse

image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')

# Compute PSNR
psnr_value = psnr(image1, image2)

# Compute SSIM
ssim_value = ssim(image1, image2)

# Compute MSE
mse_value = mse(image1, image2)

print(f"PSNR: {psnr_value} dB")
print(f"SSIM: {ssim_value}")
print(f"MSE: {mse_value}")

