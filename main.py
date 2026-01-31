import cv2
import os
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