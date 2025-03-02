import cv2
import numpy as np
import glob
import os
import shutil

# Remove black corners from image

def remove_black_borders(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        return img[y:y+h, x:x+w]
    return img


# Storing images with keypoints just for reference

def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
# Matching similarities in images

def draw_matches(img1, kp1, img2, kp2, matches):
    #Draw matches between two images
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                          matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                          flags=cv2.DrawMatchesFlags_DEFAULT)

# Extracting keypoints using SIFT

def extract_keypoints(images):
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
    keypoints_descriptors = []
    
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints_descriptors.append((kp, des))
        
        # Save image with keypoints
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"image_{i+1}_with_keypoints.jpg"), 
                    draw_keypoints(img.copy(), kp))
        
    return keypoints_descriptors

def stitch_images_sequential(images, keypoints_descriptors):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    
    if len(images) < 2:
        return images[0]
    
    result = images[0]
    
    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)}")
        
        img1 = result
        img2 = images[i]
        
        # Recompute keypoints for current result
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        
        # Get keypoints for the next image
        kp2, des2 = keypoints_descriptors[i]
        
        # Match features
        matches = matcher.knnMatch(des2, des1, k=2)
        
        # Filter good matches
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        # Save match visualization
        match_img = draw_matches(img2, kp2, img1, kp1, good_matches)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"matches_{i}_to_{i+1}.jpg"), match_img)

        # Not enough matches for stiching
        
        if len(good_matches) < 10:
            continue
        
        # Get matching points
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=2000)

        # Could not find Homography so skipping the image
        if H is None:
            continue
        
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Find transformed corners of img2
        corners = np.array([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        corners_transformed = cv2.perspectiveTransform(corners, H)
        
        # Calculate canvas size
        [xmin, ymin] = np.int32(corners_transformed.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners_transformed.max(axis=0).ravel() + 0.5)
        
        # Handle negative offsets
        xmin_offset = abs(min(xmin, 0))
        ymin_offset = abs(min(ymin, 0))
        
        # Create translation matrix
        translation_matrix = np.array([
            [1, 0, xmin_offset],
            [0, 1, ymin_offset],
            [0, 0, 1]
        ])
        
        # Apply translation to homography
        H_adjusted = translation_matrix @ H
        
        # Create panorama canvas
        panorama_width = max(xmax + xmin_offset, w1 + xmin_offset) + 100
        panorama_height = max(ymax + ymin_offset, h1 + ymin_offset) + 100
        
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # Warp second image
        warped_img = cv2.warpPerspective(img2, H_adjusted, (panorama_width, panorama_height))
        
        # Create masks
        warped_mask = cv2.warpPerspective(np.ones((h2, w2), dtype=np.uint8) * 255, 
                                          H_adjusted, (panorama_width, panorama_height))
        
        # Place first image
        panorama[ymin_offset:ymin_offset+h1, xmin_offset:xmin_offset+w1] = img1
        
        # Create first image mask
        first_image_mask = np.zeros((panorama_height, panorama_width), dtype=np.uint8)
        first_image_mask[ymin_offset:ymin_offset+h1, xmin_offset:xmin_offset+w1] = 255
        
        # Find overlap
        overlap = cv2.bitwise_and(first_image_mask, warped_mask)
        
        # Create weight maps
        weight_map1 = cv2.distanceTransform(first_image_mask, cv2.DIST_L2, 3)
        weight_map2 = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)
        
        cv2.normalize(weight_map1, weight_map1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(weight_map2, weight_map2, 0, 1, cv2.NORM_MINMAX)
        
        # Blend overlapping regions
        for y in range(panorama_height):
            for x in range(panorama_width):
                if overlap[y, x] > 0:
                    w1 = weight_map1[y, x]
                    w2 = weight_map2[y, x]
                    weight_sum = w1 + w2
                    
                    if weight_sum > 0:
                        w1 /= weight_sum
                        w2 /= weight_sum
                        
                        for c in range(3):
                            panorama[y, x, c] = np.uint8(
                                panorama[y, x, c] * w1 + 
                                warped_img[y, x, c] * w2
                            )
                elif warped_mask[y, x] > 0:
                    panorama[y, x] = warped_img[y, x]
        
        # Trim black borders
        result = remove_black_borders(panorama)
    
    return result

def stitch_images(images):
    keypoints_descriptors = extract_keypoints(images)
    return stitch_images_sequential(images, keypoints_descriptors)

# Define the output directory
OUTPUT_DIR = "Stitching_outputs"

# Create or refresh the output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Load images
image_files = sorted(glob.glob("/home/shannon/Sem_6/VR/Assignment1/Stiching1/*.jpg"))
images = []
for f in image_files:
    img = cv2.imread(f)
    if img is not None:
        images.append(img)
    else:
        print(f"Warning: Could not load image {f}")

if len(images) < 2:
    print("Not enough images to stitch.")
else:
    # Resize images if they're too large
    max_width = 1200
    for i in range(len(images)):
        h, w = images[i].shape[:2]
        if w > max_width:
            scale = max_width / w
            images[i] = cv2.resize(images[i], None, fx=scale, fy=scale)
    
    # Stitch images
    panorama = stitch_images(images)
    
    # Save the panorama
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_panorama.jpg"), panorama)
    
    print(f"All outputs have been saved to the '{OUTPUT_DIR}' directory")