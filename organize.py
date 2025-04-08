import os
import shutil
import argparse
import random
import numpy as np
from PIL import Image
import cv2

def augment_image(image_path, output_path, augmentation_type):
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return False
    
    
    if augmentation_type == "rotate90":
        augmented = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif augmentation_type == "rotate180":
        augmented = cv2.rotate(img, cv2.ROTATE_180)
    elif augmentation_type == "rotate270":
        augmented = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif augmentation_type == "flip_h":
        augmented = cv2.flip(img, 1)  
    elif augmentation_type == "flip_v":
        augmented = cv2.flip(img, 0)  
    elif augmentation_type == "brightness":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)  
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif augmentation_type == "contrast":
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        augmented = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return False
    
    
    cv2.imwrite(output_path, augmented)
    return True

def process_files(files, source_dir, target_dir, output_subdir, output_dir, augment, augmentations):
    count = 0
    
    
    for file in files:
        
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        
        
        shutil.copy(source_path, os.path.join(output_dir, output_subdir, "A", file))
        shutil.copy(target_path, os.path.join(output_dir, output_subdir, "B", file))
        count += 1
        
        
        if augment:
            filename, extension = os.path.splitext(file)
            
            for aug_type in augmentations:
                aug_filename = f"{filename}_{aug_type}{extension}"
                
                
                source_aug_path = os.path.join(output_dir, output_subdir, "A", aug_filename)
                source_success = augment_image(source_path, source_aug_path, aug_type)
                
                
                target_aug_path = os.path.join(output_dir, output_subdir, "B", aug_filename)
                target_success = augment_image(target_path, target_aug_path, aug_type)
                
                if source_success and target_success:
                    count += 1
    
    return count

def organize_dataset(source_dir, target_dir, output_dir, train_split=0.8, augment=True, seed=42):
    random.seed(seed)
    
    
    os.makedirs(os.path.join(output_dir, "train", "A"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "B"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "A"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "B"), exist_ok=True)
    
    
    augmentations = [
        "rotate90", "rotate180", "rotate270", 
        "flip_h", "flip_v"
    ]
    
    
    source_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))]
    
    
    target_files = os.listdir(target_dir)
    
    
    common_files = []
    for file in source_files:
        if file in target_files:
            common_files.append(file)
        else:
            print(f"Warning: {file} exists in source but not in target directory")
    
    
    for file in target_files:
        if file not in source_files and file.endswith(('.jpg', '.jpeg', '.png', '.tif')):
            print(f"Warning: {file} exists in target but not in source directory")
    
    print(f"Found {len(common_files)} matching files between source and target")
    
    
    random.shuffle(common_files)
    
    
    split_idx = int(len(common_files) * train_split)
    train_files = common_files[:split_idx]
    val_files = common_files[split_idx:]
    
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    
    
    train_count = process_files(
        train_files, source_dir, target_dir, 
        "train", output_dir, augment, augmentations
    )
    
    
    val_count = process_files(
        val_files, source_dir, target_dir, 
        "val", output_dir, augment, augmentations
    )
    
    print(f"Dataset organized successfully in {output_dir}")
    print("Structure:")
    print(f"  train/A: {train_count} files (source, including augmentations)")
    print(f"  train/B: {train_count} files (target, including augmentations)")
    print(f"  val/A: {val_count} files (source, including augmentations)")
    print(f"  val/B: {val_count} files (target, including augmentations)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize source and target files with augmentations")
    
    parser.add_argument("--source_dir", required=True, help="Directory containing source (ground truth) images")
    parser.add_argument("--target_dir", required=True, help="Directory containing target (prediction) images")
    parser.add_argument("--output_dir", required=True, help="Directory to create organized dataset")
    parser.add_argument("--train_split", type=float, default=0.8, help="Percentage of data to use for training (default: 0.8)")
    parser.add_argument("--no_augment", action="store_true", help="Disable augmentations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    organize_dataset(
        args.source_dir, 
        args.target_dir, 
        args.output_dir, 
        args.train_split, 
        not args.no_augment,
        args.seed
    )