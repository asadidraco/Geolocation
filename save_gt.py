import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import re
from sklearn.cluster import KMeans

def create_colormap(n_classes):
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    return colors

def apply_colormap(mask, colors):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(1, len(colors)):  
        colored_mask[mask == class_idx] = colors[class_idx]
    return colored_mask

def match_coco_image(input_filename, coco_images):
    
    base_name = os.path.splitext(input_filename)[0]
    
    
    
    for img in coco_images:
        if img['file_name'] == input_filename:
            return img
    
    
    for img in coco_images:
        coco_base = os.path.splitext(img['file_name'])[0]
        if coco_base == base_name:
            return img
    
    
    input_numbers = re.findall(r'\d+', base_name)
    if input_numbers:
        last_number = input_numbers[-1]
        
        for img in coco_images:
            coco_base = os.path.splitext(img['file_name'])[0]
            coco_numbers = re.findall(r'\d+', coco_base)
            
            if coco_numbers and coco_numbers[-1] == last_number:
                return img
    
    
    if input_numbers:
        for img in coco_images:
            coco_base = os.path.splitext(img['file_name'])[0]
            
            for num in input_numbers:
                if coco_base.endswith(num):
                    return img
    
    
    return None

def create_mask_from_coco_annotation(image_id, annotations, image_height, image_width):
    
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    
    
    for ann in image_annotations:
        category_id = ann['category_id']
        
        
        if 'segmentation' in ann and isinstance(ann['segmentation'], list):
            for polygon in ann['segmentation']:
                
                poly = np.array(polygon).reshape(-1, 2).astype(np.int32)
                
                cv2.fillPoly(mask, [poly], category_id)
    
    
    filled_mask = mask.copy()
    kernel = np.ones((5, 5), np.uint8)
    
    
    all_segments = np.zeros_like(mask, dtype=bool)
    for category_id in range(1, 6):  
        class_mask = (mask == category_id)
        all_segments = np.logical_or(all_segments, class_mask)
    
    
    gaps = np.logical_not(all_segments)
    
    
    if np.any(gaps):
        
        for category_id in range(1, 6):  
            if not np.any(mask == category_id):
                continue
                
            binary = np.zeros_like(mask, dtype=np.uint8)
            binary[mask == category_id] = 1
            
            dilated = cv2.dilate(binary, kernel, iterations=3)
            
            dilated_gaps = np.logical_and(dilated.astype(bool), gaps)
            filled_mask[dilated_gaps] = category_id
            
            
            gaps = np.logical_and(gaps, np.logical_not(dilated_gaps))
    
    return filled_mask

def fill_gaps_with_clustering(mask, num_classes):
    
    gaps = (mask == 0)
    
    
    if not np.any(gaps):
        return mask
    
    
    filled_mask = mask.copy()
    
    
    gap_coords = np.argwhere(gaps)
    
    
    if len(gap_coords) > 10000:
        np.random.seed(42)
        indices = np.random.choice(len(gap_coords), 10000, replace=False)
        gap_coords = gap_coords[indices]
    
    
    class_pixels = {}
    for class_id in range(1, num_classes):
        class_mask = (mask == class_id)
        if np.any(class_mask):
            class_pixels[class_id] = np.argwhere(class_mask)
    
    
    for y, x in gap_coords:
        min_dist = float('inf')
        nearest_class = 0
        
        for class_id, pixels in class_pixels.items():
            
            distances = np.sqrt(np.sum((pixels - np.array([y, x]))**2, axis=1))
            min_class_dist = np.min(distances)
            
            if min_class_dist < min_dist:
                min_dist = min_class_dist
                nearest_class = class_id
        
        
        if nearest_class > 0:
            filled_mask[y, x] = nearest_class
    
    
    remaining_gaps = (filled_mask == 0)
    if np.any(remaining_gaps):
        
        num_labels, labels = cv2.connectedComponents(remaining_gaps.astype(np.uint8))
        
        
        for label in range(1, num_labels):
            component = (labels == label)
            
            
            dilated_component = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8))
            border = dilated_component.astype(bool) & ~component
            
            
            border_classes = filled_mask[border]
            if len(border_classes) > 0:
                
                border_classes = border_classes[border_classes > 0]
                if len(border_classes) > 0:
                    most_common = np.bincount(border_classes).argmax()
                    filled_mask[component] = most_common
    
    
    for i in range(3):  
        for class_id in range(1, num_classes):
            class_mask = (filled_mask == class_id)
            if not np.any(class_mask):
                continue
            
            
            kernel = np.ones((3, 3), np.uint8)
            mask_closed = cv2.morphologyEx(class_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            
            update_mask = (mask_closed == 1) & (mask == 0)
            filled_mask[update_mask] = class_id
    
    return filled_mask

def generate_colored_gt(args):
    
    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file {args.annotations} not found")
        return
    
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory {args.images_dir} not found")
        return
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    print(f"Loading COCO annotations from {args.annotations}")
    with open(args.annotations, 'r') as f:
        coco_data = json.load(f)
    
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    num_classes = len(categories) + 1  
    
    
    input_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        input_files.extend([f for f in os.listdir(args.input_dir) 
                            if f.lower().endswith(ext)])
    
    if not input_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    
    colors = create_colormap(num_classes)
    
    
    print(f"Input directory: {args.input_dir}")
    print(f"Number of input files: {len(input_files)}")
    print(f"Number of COCO images: {len(coco_data['images'])}")
    print(f"Number of categories: {len(categories)}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    
    
    matched_files = []
    unmatched_files = []
    
    for input_filename in tqdm(input_files, desc="Matching images"):
        matched_image = match_coco_image(input_filename, coco_data['images'])
        if matched_image:
            matched_files.append((input_filename, matched_image))
        else:
            unmatched_files.append(input_filename)
    
    print(f"Matched {len(matched_files)} out of {len(input_files)} input files")
    
    
    for idx, (input_filename, coco_image) in enumerate(matched_files):
        print(f"Processing file {idx+1}/{len(matched_files)}: {input_filename}")
        
        
        image_id = coco_image['id']
        image_height = coco_image['height']
        image_width = coco_image['width']
        
        
        mask = create_mask_from_coco_annotation(
            image_id, 
            coco_data['annotations'], 
            image_height, 
            image_width
        )
        
        
        resized_mask = cv2.resize(
            mask, 
            (args.target_size, args.target_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        
        print(f"  Filling remaining gaps with clustering...")
        filled_mask = fill_gaps_with_clustering(resized_mask, num_classes)
        
        
        colored_mask = apply_colormap(filled_mask, colors)
        
        
        base_filename = os.path.splitext(input_filename)[0]
        output_path = os.path.join(args.output_dir, f"{base_filename}.png")
        cv2.imwrite(output_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        
        
        if args.save_comparison:
            comparison_dir = os.path.join(args.output_dir, "comparisons")
            os.makedirs(comparison_dir, exist_ok=True)
            
            
            original_colored = apply_colormap(resized_mask, colors)
            filled_colored = apply_colormap(filled_mask, colors)
            
            
            comparison = np.hstack([original_colored, filled_colored])
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w = original_colored.shape[:2]
            comp_img = np.zeros((h + 30, w * 2, 3), dtype=np.uint8)
            comp_img[30:, :] = comparison
            
            cv2.putText(comp_img, "Before Clustering", (10, 20), font, 0.7, (255, 255, 255), 2)
            cv2.putText(comp_img, "After Clustering", (w + 10, 20), font, 0.7, (255, 255, 255), 2)
            
            
            cv2.imwrite(os.path.join(comparison_dir, f"{base_filename}_comparison.png"), 
                        cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))
    
    
    legend_size = 100
    legend_img = np.zeros((legend_size * len(categories), legend_size, 3), dtype=np.uint8)
    
    for i, (cat_id, cat_name) in enumerate(categories.items()):
        legend_img[i*legend_size:(i+1)*legend_size, :, :] = colors[cat_id]
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_img = np.ones((legend_size, legend_size*4, 3), dtype=np.uint8) * 255
        cv2.putText(text_img, f"{cat_id}: {cat_name}", (10, 50), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        
        combined = np.hstack([legend_img[i*legend_size:(i+1)*legend_size, :, :], text_img])
        
        
        
    
    print(f"Colored GT masks saved to {os.path.abspath(args.output_dir)}")
    print(f"Total masks generated: {len(matched_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate colored ground truth masks from COCO annotations with advanced gap filling")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing input images to generate GT for")
    parser.add_argument("--annotations", type=str, default="annotations.json", 
                        help="Path to COCO annotations file")
    parser.add_argument("--images_dir", type=str, default="saiod", 
                        help="Directory containing original images referenced in annotations")
    parser.add_argument("--output_dir", type=str, default="Source", 
                        help="Directory to save colored ground truth masks")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target size for output masks (square)")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save before/after comparison images")
    
    args = parser.parse_args()
    
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
    else:
        generate_colored_gt(args)