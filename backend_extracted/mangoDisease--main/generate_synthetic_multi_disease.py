import os
import cv2
import numpy as np
import random
from pathlib import Path
from glob import glob
from tqdm import tqdm

def create_synthetic_dataset(base_dir, output_dir, target_count=2000):
    os.makedirs(output_dir, exist_ok=True)
    
    # In many Github repos, images are grouped by folders:
    # e.g., base_dir/Anthracnose, base_dir/Healthy, etc.
    # We will find all jpg/png files.
    all_images = glob(os.path.join(base_dir, '**', '*.*'), recursive=True)
    all_images = [img for img in all_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(all_images) == 0:
        print(f"Error: No images found in {base_dir}")
        return

    # Group images by their parent directory name (class)
    classes = {}
    for img_path in all_images:
        cls_name = os.path.basename(os.path.dirname(img_path))
        if cls_name not in classes:
            classes[cls_name] = []
        classes[cls_name].append(img_path)
    
    class_names = list(classes.keys())
    print(f"Found classes: {class_names}")

    if len(class_names) < 2:
        print("Need at least 2 classes to merge diseases!")
        return

    generated_count = 0
    print(f"Generating {target_count} synthetic multi-disease images...")
    
    for i in tqdm(range(target_count)):
        # Randomly select two distinct classes
        cls1, cls2 = random.sample(class_names, 2)
        
        # Randomly select an image from each class
        img1_path = random.choice(classes[cls1])
        img2_path = random.choice(classes[cls2])
        
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            continue
            
        # Resize img2 to match img1 dimensions if they differ
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # We want to extract disease spots from img2 and paste them on img1.
        # Simple heuristic: disease spots are often non-green. 
        # Convert img2 to HSV to create a mask for non-green regions.
        hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Define range for green color in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Mask out green (healthy part)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # The disease mask is the inverse of the green mask
        # We also want to ignore pure black/background, so we create a background mask
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        disease_mask = cv2.bitwise_not(green_mask)
        # Only keep disease parts that are inside the leaf (not background)
        disease_mask = cv2.bitwise_and(disease_mask, bg_mask)
        
        # Apply morphology to smooth the mask
        kernel = np.ones((5,5), np.uint8)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
        
        # Blend the images: where disease_mask is > 0, take img2, else take img1
        # To make it look more natural, we use seamlessClone if the mask is valid
        # Find contours of the disease mask to find bounding rects
        contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blended = img1.copy()
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100: # filter out tiny noise
                x, y, w, h = cv2.boundingRect(cnt)
                center = (x + w//2, y + h//2)
                
                # Check if the center is within bounds
                if center[0] > 0 and center[0] < blended.shape[1] and center[1] > 0 and center[1] < blended.shape[0]:
                    try:
                        # Extract the region mask for this contour
                        cnt_mask = np.zeros_like(disease_mask)
                        cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                        
                        # Poisson blending for seamless integration
                        blended = cv2.seamlessClone(img2, blended, cnt_mask, center, cv2.NORMAL_CLONE)
                    except Exception as e:
                        # If seamless clone fails, fallback to simple overlay for this patch
                        roi1 = blended[y:y+h, x:x+w]
                        roi2 = img2[y:y+h, x:x+w]
                        roi_mask = cnt_mask[y:y+h, x:x+w]
                        
                        for c in range(3):
                            roi1[:,:,c] = np.where(roi_mask > 0, roi2[:,:,c], roi1[:,:,c])
                        blended[y:y+h, x:x+w] = roi1

        # Save the result
        filename = f"synthetic_{cls1}_{cls2}_{generated_count}.jpg"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, blended)
        generated_count += 1

if __name__ == "__main__":
    # Adjust this path based on where the github repo stores the images
    base_dataset_path = "/Users/abhinavsahu/Downloads/projects/mangoDisease/base_dataset"
    output_dataset_path = "/Users/abhinavsahu/Downloads/projects/mangoDisease/synthetic_multi_disease"
    
    create_synthetic_dataset(base_dataset_path, output_dataset_path, target_count=2000)
    print("Done! Check the output folder.")
