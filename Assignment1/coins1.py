import cv2
import numpy as np
import os
import shutil

# Converts image to grayscale and then blurs the image using Gaussian blur.


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    return image, gray, blurred

# Canny Edge detector
def apply_canny(blurred):
    return cv2.Canny(blurred, 50, 150)

# Sobel edge detection
def apply_sobel(blurred):
    blurred = cv2.GaussianBlur(blurred, (17, 17), 2.6)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel_im = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y)
    ret, thresh1 = cv2.threshold(sobel_im, 120, 255, cv2.THRESH_BINARY)
    return thresh1

# Laplacian edge detector
def apply_laplacian(blurred):
    blurred = cv2.GaussianBlur(blurred, (15, 15), 2.5)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
    return cv2.convertScaleAbs(laplacian)

# Draw contours
def detect_contours(edges, image):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output, contours

# Save images
def save(image, name):
    output_folder = "Outputs_Q1"
    os.makedirs(output_folder, exist_ok=True)
    
    # cv2.imshow(name, image)
    cv2.imwrite(f"{output_folder}/{name}.jpg", image)


def segment_and_save_coins(image_path, output_dir="Output_Q1_p2"):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create output directories
    coin_output_dir = os.path.join(output_dir, "Coins")

    # Check if the directory exists and delete it
    if os.path.exists(coin_output_dir):
        shutil.rmtree(coin_output_dir)

    # Recreate the directory after deletion
    os.makedirs(coin_output_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
   

    # Adaptive thresholding to segment coins
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing to clean up noise
    kernel = np.ones((5, 5), np.uint8)  # Slightly larger kernel for better coin separation
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Masked image (for segmented coins overlay)
    masked_image = np.zeros_like(image)

    count = 1

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Reject very small or very large objects
        if area < 5000 or area > 35000:  
            continue

        # Calculate perimeter and circularity to filter out irregular shapes
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        if circularity < 0.7:  # Perfect circles have circularity ~1, setting threshold >0.7
            continue

        # Fit a minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # Create a mask for the coin
        mask = np.zeros_like(gray)
        cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)

        # Extract the coin using the circular mask
        coin = cv2.bitwise_and(image, image, mask=mask)

        # Crop precisely around the circle
        x, y, w, h = center[0] - radius, center[1] - radius, 2 * radius, 2 * radius
        coin_cropped = coin[y:y+h, x:x+w]

        # Save the extracted coin
        coin_filename = os.path.join(coin_output_dir, f"coin_{count}.png")
        cv2.imwrite(coin_filename, coin_cropped)
        count = count + 1

        # Overlay detected coins on masked image
        cv2.circle(masked_image, center, radius, (0, 255, 0), thickness=-1)  # Green filled circle

    # Save the masked segmented image
    masked_image_path = os.path.join(output_dir, "segmented_coins.png")
    cv2.imwrite(masked_image_path, masked_image)

    print(f"Segmented coins saved in '{coin_output_dir}'")
    print(f"Masked segmented image saved at '{masked_image_path}'")


def count_coins(segmented_image_path):
    #Counts the number of coins in the segmented image using contour detection.
    
    # Load the segmented image
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Image not found!")
        return 0

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    # Apply thresholding to create a binary mask
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove noise
    min_area = 5000  # Adjust according to expected coin size
    max_area = 35000
    coin_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    print(f"Detected {len(coin_contours)} coins.")
    
    return len(coin_contours)



def main():
    image_path = os.path.join("Images_Q1", "Image2.jpeg")  # Form the path to the image
    
    # Preprocess image
    image, gray, blurred = preprocess_image(image_path)

    # Apply edge detection
    canny_edges = apply_canny(blurred)
    sobel_edges = apply_sobel(gray)
    laplacian_edges = apply_laplacian(gray)

    # Detect and outline coins using Canny edges
    detected_image, contours = detect_contours(canny_edges, image)

    # Save and display results
    save(canny_edges, "Canny_Edges")
    save(sobel_edges, "Sobel_Edges")
    save(laplacian_edges, "Laplacian_Edges")
    save(detected_image, "Detected_Coins")

    # print(f"Detected {len(contours)} coins. Output saved in 'Outputs_Q1'.") # One more method to count coins
    segment_and_save_coins(image_path)

    segmented_image_path = "Output_Q1_p2/segmented_coins.png"  # Path to the saved segmented image
    num_coins = count_coins(segmented_image_path)
    print(f"Number of coins detected: {num_coins}")


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()