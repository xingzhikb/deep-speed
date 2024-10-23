import os
import cv2
import numpy as np
import random

def detect_green_rectangles(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} could not be read.")
        return [], None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 20 and h > 20:
                bboxes.append([x, y, w, h])
    
    return bboxes, image

def test_detection(image_dir, output_dir, K):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    
    for i, image_file in enumerate(image_files[:K]):
        image_path = os.path.join(image_dir, image_file)
        bboxes, image = detect_green_rectangles(image_path)
        
        if image is None:
            continue
        
        for bbox in bboxes:
            x, y, w, h = bbox
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        output_path = os.path.join(output_dir, f"test_output_{i+1}.png")
        cv2.imwrite(output_path, image)
        
        print(f"Processed {image_file}: {len(bboxes)} green rectangles detected")

    print(f"Processed {K} images. Results saved in {output_dir}")

# Example usage
image_dir = '/workspace/mimic_cxr'
test_output_dir = '/workspace/test_output'
K = 10  # Number of images to test

# Run the test function
test_detection(image_dir, test_output_dir, K)