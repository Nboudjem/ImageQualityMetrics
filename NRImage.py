import cv2
import pyiqa  # PyIQA supports multiple IQA metrics


# Load image (grayscale for better quality analysis)
image_path = 'image.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure image is in proper format
if image is None:
    raise ValueError("Image not found or invalid format")

# Initialize PyIQA models
brisque_model = pyiqa.create_metric('brisque')  # Blind/Referenceless IQA
niqe_model = pyiqa.create_metric('niqe')  # Naturalness Image Quality Evaluator
piqe_model = pyiqa.create_metric('piqe')  # Perception-based IQA

# Convert image to PyIQA tensor format
image_tensor = pyiqa.utils.to_tensor(image).unsqueeze(0)

# Compute NR-IQA scores
brisque_score = brisque_model(image_tensor).item()
niqe_score = niqe_model(image_tensor).item()
piqe_score = piqe_model(image_tensor).item()

# Print results
print(f"BRISQUE Score: {brisque_score:.2f} (Lower is better)")
print(f"NIQE Score: {niqe_score:.2f} (Lower is better)")
print(f"PIQE Score: {piqe_score:.2f} (Lower is better)")
