import os
import json
import cv2
import numpy as np

def process_output_jsonl(output_jsonl, image_dir):
    processed_images_dir = os.path.join(image_dir, "processed_images")
    if not os.path.exists(processed_images_dir):
        os.makedirs(processed_images_dir)
    
    with open(output_jsonl, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())
            file_name = data.get('file_name')

            if not file_name:
                continue
            
            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Image {image_path} could not be read.")
                continue
            
            # Convert image to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define range for green color in HSV
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([80, 255, 255])
            
            # Create a mask for green color
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bboxes = []
            for contour in contours:
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If the polygon has 4 vertices, it's likely a rectangle
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    # Only consider rectangles of a certain size
                    if w > 20 and h > 20:
                        bboxes.append([x, y, w, h])
            
            if len(bboxes) > 0:
                for i, bbox in enumerate(bboxes):
                    copy_image_name = f"{os.path.splitext(file_name)[0]}_{i+1}.png"
                    copy_image_path = os.path.join(processed_images_dir, copy_image_name)
                    
                    # Draw the bounding box on the image
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Save a copy of the image with the bounding box
                    cv2.imwrite(copy_image_path, image)
                    
                    new_data = data.copy()
                    new_data['file_name'] = os.path.relpath(copy_image_path, image_dir)
                    new_data['bbox'] = bbox
                    
                    processed_output_file = os.path.join(processed_images_dir, 'processed_output.jsonl')
                    with open(processed_output_file, 'a') as outfile:
                        json.dump(new_data, outfile)
                        outfile.write('\n')
            else:
                print(f"No green rectangles detected in {file_name}")

# Example usage
output_jsonl = '/workspace/mimic_cxr/output.jsonl'
image_dir = '/workspace/mimic_cxr'

process_output_jsonl(output_jsonl, image_dir)