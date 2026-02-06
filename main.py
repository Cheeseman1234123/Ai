import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Plain Image
image = cv2.imread('images.jpg')
cv2.namedWindow('Loaded Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Loaded Image', 500, 500)
cv2.imshow('Loaded Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Image Dimension: {image.shape}")

# RGB Image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("RGB Image")
plt.show()

# Gray Scale Image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image)
plt.title("Gray Image")
plt.show()

# Cropped Image
cropped_image = image[100:300, 200:400]
cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
plt.imshow(cropped_rgb)
plt.title("Cropped Region")
plt.show()

# Rotated Image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(h, w) = image.shape[:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
plt.imshow(rotated_rgb)
plt.title("Rotated RGB")
plt.show()

# Brightened Image
brightness_matrix = np.ones(image.shape, dtype="uint8") * 50
brighter = cv2.add(image, brightness_matrix)
brighter_rgb = cv2.cvtColor(brighter, cv2.COLOR_BGR2RGB)
plt.imshow(brighter_rgb)
plt.title("Brighter RGB")
plt.show()

# Image Size Changes
image_path = "images.jpg"
output_dir = "resized_images"
os.makedirs(output_dir, exist_ok=True)
sizes = {
    "small": {320, 240},
    "medium": {640, 480},
    "large": {1024, 768}
}
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
for name, (width, height) in sizes.items():
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    window_name = f"{name} ({width}x{height})"
    cv2.imshow(window_name, resized)
    output_path = os.path.join(output_dir, f"{name}.jpg")
    cv2.imwrite(output_path, resized)
cv2.waitKey(0)
cv2.destroyAllWindows()