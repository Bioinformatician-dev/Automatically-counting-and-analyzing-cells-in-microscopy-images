import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image_gray

# Function to apply Gaussian blur and thresholding
def preprocess_image(image_gray):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return morph_image

# Function to find and count contours
def count_cells(morph_image):
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to visualize results
def visualize_results(image, contours):
    output_image = image.copy()
    cell_count = 0
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes
            cell_count += 1
    
    print(f'Total number of cells detected: {cell_count}')
    
    # Display the output image with bounding boxes
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Cell Counting with Bounding Boxes')
    plt.axis('off')
    plt.show()

# Main function
def main(image_path):
    image, image_gray = load_and_preprocess_image(image_path)
    morph_image = preprocess_image(image_gray)
    contours = count_cells(morph_image)
    visualize_results(image, contours)

# Replace 'microscopy_image.jpg' with your image path
if __name__ == "__main__":
    main('microscopy_image.jpg')
