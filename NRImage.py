import cv2
import pyiqa  # PyIQA supports multiple IQA metrics
import ssl
import torchvision.transforms as transforms
ssl._create_default_https_context = ssl._create_unverified_context


'''
No-Reference (NR) image quality assessment methods evaluate an image's quality without
requiring a reference image. These metrics are useful in real-world scenarios where 
only the degraded image is available.
 
Understanding the Scores
BRISQUE (0 - 100)

Lower values indicate better quality.
BRISQUE uses handcrafted features trained on natural scene statistics.
NIQE (1 - 20+)

Lower values indicate a more natural-looking image.
NIQE is model-free and doesn't require a database for training.
PIQE (0 - 100)

Lower values indicate a better perceptual quality.
PIQE is more focused on block-level distortion in images.
'''


# Load image (grayscale for better quality analysis)
image_path = 'C:/Users/User/Desktop/ImageQuality/chest.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure image is in proper format
if image is None:
    raise ValueError("Image not found or invalid format")

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

# Print results
print(f"BRISQUE Score: {brisque_score:.2f} (Lower is better)")
print(f"NIQE Score: {niqe_score:.2f} (Lower is better)")
print(f"PIQE Score: {piqe_score:.2f} (Lower is better)")
