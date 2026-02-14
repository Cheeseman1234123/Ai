import cv2
import numpy as np
def apply_color_filter(image, filter_type):
    """Apply the specified color filter to the Image"""
    filtered_image = image.copy()
    if filter_type == "red_tint":
        filtered_image[:, :, 1] = 0
        filtered_image[:, :, 0] = 0
    elif filter_type == "blue_tint":
        filtered_image[:, :, 1] = 0
        filtered_image[:, :, 2] = 0
    elif filter_type == "green_tint":
        filtered_image[:, :, 0] = 0
        filtered_image[:, :, 2] = 0
    elif filter_type == "increase_red":
        filtered_image[:, :, 2] = cv2.add(filtered_image[:, :, 2], 50)
    elif filter_type == "decrease_blue":
        filtered_image[:, :, 2] = cv2.subtract(filtered_image[:, :, 0], 50)
    elif filter_type == "grayscale":
        filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == "sepis":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered_image = cv2.transform(image, kernel)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image
image_path = "images.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")
else:
    filter_type = "original"
    print("Press the following keys to apply the filters:")
    print("r - Red Tint")
    print("b - Blue Tint")
    print("g - Green Tint")
    print("i - Increase Red")
    print("d - Decrease Blue")
    print("s - Sepis")
    print("w - Grayscale")
    print("o - Original")
    print("q - Quit")
    while True: 
        filtered_image = apply_color_filter(image, filter_type)
        cv2.imshow("Filtered Image", filtered_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            filter_type = "red_tint"
        elif key == ord("b"):
           filter_type = "blue_tint"
        elif key == ord("g"):
            filter_type = "green_tint"
        elif key == ord("i"):
            filter_type = "increase_red"
        elif key == ord("d"):
            filter_type = "decrease_blue"
        elif key == ord("s"):
            filter_type = "sepis"
        elif key == ord("w"):
            filter_type = "grayscale"
        elif key == ord("o"):
            filter_type = "original"
        elif key == ord("q"):
            print("Exiting...")
            break
    cv2.destroyAllWindows()