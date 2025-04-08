import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import albumentations as A
import random
from pathlib import Path


np.random.seed(42)
random.seed(42)

class Config:
    
    IMAGE_DIR = "saiod"  
    ANNOTATION_FILE = "annotations.json"
    
    
    OUTPUT_BASE_DIR = "dataset"  
    IMAGE_SIZE = 512  
    
    
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.25
    TEST_RATIO = 0.15  

def create_mask_from_coco_annotation(annotation, image_height, image_width, categories):

    
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    
    for ann in annotation:
        category_id = ann['category_id']
        
        
        if 'segmentation' in ann and isinstance(ann['segmentation'], list):
            for polygon in ann['segmentation']:
                
                poly = np.array(polygon).reshape(-1, 2).astype(np.int32)
                
                cv2.fillPoly(mask, [poly], category_id)
    
    return mask

def split_image_and_mask(image, mask, num_parts=4):

    height, width = image.shape[:2]
    half_height, half_width = height // 2, width // 2
    
    parts = []
    
    
    regions = [
        (0, 0, half_width, half_height),
        (half_width, 0, width, half_height),
        (0, half_height, half_width, height),
        (half_width, half_height, width, height)
    ]
    
    for x1, y1, x2, y2 in regions:
        img_part = image[y1:y2, x1:x2].copy()
        mask_part = mask[y1:y2, x1:x2].copy()
        parts.append((img_part, mask_part))
    
    return parts

def apply_augmentations(image, mask):

    result = [(image.copy(), mask.copy())]
    
    
    h_flip = A.HorizontalFlip(p=1.0)
    h_flipped = h_flip(image=image, mask=mask)
    result.append((h_flipped['image'], h_flipped['mask']))
    
    
    v_flip = A.VerticalFlip(p=1.0)
    v_flipped = v_flip(image=image, mask=mask)
    result.append((v_flipped['image'], v_flipped['mask']))
    
    
    hv_flipped = h_flip(image=v_flipped['image'], mask=v_flipped['mask'])
    result.append((hv_flipped['image'], hv_flipped['mask']))
    
    return result

def save_colormap(categories, output_dir):
    n_classes = len(categories) + 1  
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    
    
    plt.figure(figsize=(10, 5))
    
    
    for i, (cat_id, cat_name) in enumerate(categories.items()):
        color = colors[cat_id][:3]  
        plt.bar(i, 1, color=color)
        plt.text(i, 0.5, f"{cat_id}: {cat_name}", 
                 ha='center', va='center', rotation=90, color='white' if sum(color) < 1.5 else 'black')
    
    plt.title("Category Color Mapping")
    plt.xlim(-0.5, len(categories) - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_colormap.png"))
    plt.close()

def prepare_dataset():

    config = Config()
    
    
    os.makedirs(config.OUTPUT_BASE_DIR, exist_ok=True)
    
    
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(config.OUTPUT_BASE_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(config.OUTPUT_BASE_DIR, split, 'masks'), exist_ok=True)
    
    
    print("Loading COCO annotations...")
    with open(config.ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)
    
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    
    with open(os.path.join(config.OUTPUT_BASE_DIR, 'categories.json'), 'w') as f:
        json.dump(categories, f, indent=2)
    
    
    save_colormap(categories, config.OUTPUT_BASE_DIR)
    
    
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    id_to_size = {img['id']: (img['height'], img['width']) for img in coco_data['images']}
    
    
    image_annotations = {}
    for annotation in coco_data['annotations']:
        img_id = annotation['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(annotation)
    
    
    image_ids = list(id_to_filename.keys())
    
    
    train_val_ids, test_ids = train_test_split(
        image_ids, 
        test_size=config.TEST_RATIO, 
        random_state=42
    )
    
    
    val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_ids, val_ids = train_test_split(
        train_val_ids, 
        test_size=val_ratio_adjusted, 
        random_state=42
    )
    
    
    split_mapping = {}
    for img_id in train_ids:
        split_mapping[img_id] = 'train'
    for img_id in val_ids:
        split_mapping[img_id] = 'val'
    for img_id in test_ids:
        split_mapping[img_id] = 'test'
    
    
    with open(os.path.join(config.OUTPUT_BASE_DIR, 'split_mapping.json'), 'w') as f:
        json.dump(split_mapping, f, indent=2)
    
    
    print("Processing original images and creating masks...")
    processed_files = {}  
    
    for split in splits:
        processed_files[split] = []
    
    for img_id, split in tqdm(split_mapping.items(), desc="Processing original images"):
        filename = id_to_filename[img_id]
        img_path = os.path.join(config.IMAGE_DIR, filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping")
            continue
        
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}, skipping")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        height, width = id_to_size[img_id]
        mask = create_mask_from_coco_annotation(
            image_annotations.get(img_id, []), 
            height, 
            width, 
            categories
        )
        
        
        resize_transform = A.Compose([
            A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)
        ])
        resized = resize_transform(image=image, mask=mask)
        image_resized, mask_resized = resized['image'], resized['mask']
        
        
        base_filename = os.path.splitext(filename)[0]
        image_output_path = os.path.join(
            config.OUTPUT_BASE_DIR, split, 'images', f"{base_filename}.png"
        )
        mask_output_path = os.path.join(
            config.OUTPUT_BASE_DIR, split, 'masks', f"{base_filename}.png"
        )
        
        cv2.imwrite(image_output_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_output_path, mask_resized)
        
        processed_files[split].append((base_filename, image_output_path, mask_output_path))
    
    
    # if processed_files['val']:
    #     print("Applying augmentations to validation set...")
    #     train_augmented_files = []
        
        
    #     for base_filename, image_path, mask_path in tqdm(processed_files['val'], desc="Basic augmentations"):
            
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            
    #         augmented_pairs = apply_augmentations(image, mask)
            
            
    #         for i, (aug_image, aug_mask) in enumerate(augmented_pairs[1:], 1):
    #             aug_suffix = {1: "h_flip", 2: "v_flip", 3: "hv_flip"}[i]
    #             aug_image_path = os.path.join(
    #                 config.OUTPUT_BASE_DIR, 'val', 'images', 
    #                 f"{base_filename}_{aug_suffix}.png"
    #             )
    #             aug_mask_path = os.path.join(
    #                 config.OUTPUT_BASE_DIR, 'val', 'masks', 
    #                 f"{base_filename}_{aug_suffix}.png"
    #             )
                
    #             cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
    #             cv2.imwrite(aug_mask_path, aug_mask)
                
    #             train_augmented_files.append((f"{base_filename}_{aug_suffix}", aug_image_path, aug_mask_path))
        
        
    #     processed_files['val'].extend(train_augmented_files)
        
        
    #     print("Splitting images into quarters and applying more augmentations...")
    #     quarters_files = []
        
        
    #     all_train_files = processed_files['val'].copy()
        
    #     for base_filename, image_path, mask_path in tqdm(all_train_files, desc="Splitting and augmenting"):
            
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            
    #         quarter_pairs = split_image_and_mask(image, mask)
            
            
    #         for q_idx, (q_image, q_mask) in enumerate(quarter_pairs):
                
    #             resize_transform = A.Compose([
    #                 A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)
    #             ])
    #             resized = resize_transform(image=q_image, mask=q_mask)
    #             q_image_resized, q_mask_resized = resized['image'], resized['mask']
                
                
    #             quarter_suffix = f"q{q_idx+1}"
    #             q_image_path = os.path.join(
    #                 config.OUTPUT_BASE_DIR, 'val', 'images',
    #                 f"{base_filename}_{quarter_suffix}.png"
    #             )
    #             q_mask_path = os.path.join(
    #                 config.OUTPUT_BASE_DIR, 'val', 'masks',
    #                 f"{base_filename}_{quarter_suffix}.png"
    #             )
                
    #             cv2.imwrite(q_image_path, cv2.cvtColor(q_image_resized, cv2.COLOR_RGB2BGR))
    #             cv2.imwrite(q_mask_path, q_mask_resized)
                
    #             quarters_files.append((f"{base_filename}_{quarter_suffix}", q_image_path, q_mask_path))
                
                
    #             quarter_augmented_pairs = apply_augmentations(q_image_resized, q_mask_resized)
                
                
    #             for i, (aug_image, aug_mask) in enumerate(quarter_augmented_pairs[1:], 1):
    #                 aug_suffix = {1: "h_flip", 2: "v_flip", 3: "hv_flip"}[i]
    #                 aug_image_path = os.path.join(
    #                     config.OUTPUT_BASE_DIR, 'val', 'images',
    #                     f"{base_filename}_{quarter_suffix}_{aug_suffix}.png"
    #                 )
    #                 aug_mask_path = os.path.join(
    #                     config.OUTPUT_BASE_DIR, 'val', 'masks',
    #                     f"{base_filename}_{quarter_suffix}_{aug_suffix}.png"
    #                 )
                    
    #                 cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
    #                 cv2.imwrite(aug_mask_path, aug_mask)
                    
    #                 quarters_files.append((f"{base_filename}_{quarter_suffix}_{aug_suffix}", 
    #                                        aug_image_path, aug_mask_path))
        
        
    #     processed_files['val'].extend(quarters_files)
    
    if processed_files['train']:
        print("Applying augmentations to training set...")
        train_augmented_files = []
        
        
        for base_filename, image_path, mask_path in tqdm(processed_files['train'], desc="Basic augmentations"):
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            
            augmented_pairs = apply_augmentations(image, mask)
            
            
            for i, (aug_image, aug_mask) in enumerate(augmented_pairs[1:], 1):
                aug_suffix = {1: "h_flip", 2: "v_flip", 3: "hv_flip"}[i]
                aug_image_path = os.path.join(
                    config.OUTPUT_BASE_DIR, 'train', 'images', 
                    f"{base_filename}_{aug_suffix}.png"
                )
                aug_mask_path = os.path.join(
                    config.OUTPUT_BASE_DIR, 'train', 'masks', 
                    f"{base_filename}_{aug_suffix}.png"
                )
                
                cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(aug_mask_path, aug_mask)
                
                train_augmented_files.append((f"{base_filename}_{aug_suffix}", aug_image_path, aug_mask_path))
        
        
        processed_files['train'].extend(train_augmented_files)
        
        
        print("Splitting images into quarters and applying more augmentations...")
        quarters_files = []
        
        
        all_train_files = processed_files['train'].copy()
        
        for base_filename, image_path, mask_path in tqdm(all_train_files, desc="Splitting and augmenting"):
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            
            quarter_pairs = split_image_and_mask(image, mask)
            
            
            for q_idx, (q_image, q_mask) in enumerate(quarter_pairs):
                
                resize_transform = A.Compose([
                    A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)
                ])
                resized = resize_transform(image=q_image, mask=q_mask)
                q_image_resized, q_mask_resized = resized['image'], resized['mask']
                
                
                quarter_suffix = f"q{q_idx+1}"
                q_image_path = os.path.join(
                    config.OUTPUT_BASE_DIR, 'train', 'images',
                    f"{base_filename}_{quarter_suffix}.png"
                )
                q_mask_path = os.path.join(
                    config.OUTPUT_BASE_DIR, 'train', 'masks',
                    f"{base_filename}_{quarter_suffix}.png"
                )
                
                cv2.imwrite(q_image_path, cv2.cvtColor(q_image_resized, cv2.COLOR_RGB2BGR))
                cv2.imwrite(q_mask_path, q_mask_resized)
                
                quarters_files.append((f"{base_filename}_{quarter_suffix}", q_image_path, q_mask_path))
                
                
                quarter_augmented_pairs = apply_augmentations(q_image_resized, q_mask_resized)
                
                
                for i, (aug_image, aug_mask) in enumerate(quarter_augmented_pairs[1:], 1):
                    aug_suffix = {1: "h_flip", 2: "v_flip", 3: "hv_flip"}[i]
                    aug_image_path = os.path.join(
                        config.OUTPUT_BASE_DIR, 'train', 'images',
                        f"{base_filename}_{quarter_suffix}_{aug_suffix}.png"
                    )
                    aug_mask_path = os.path.join(
                        config.OUTPUT_BASE_DIR, 'train', 'masks',
                        f"{base_filename}_{quarter_suffix}_{aug_suffix}.png"
                    )
                    
                    cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(aug_mask_path, aug_mask)
                    
                    quarters_files.append((f"{base_filename}_{quarter_suffix}_{aug_suffix}", 
                                           aug_image_path, aug_mask_path))
        
        
        processed_files['train'].extend(quarters_files)
    

    
    for split in splits:
        with open(os.path.join(config.OUTPUT_BASE_DIR, f'{split}_files.txt'), 'w') as f:
            for base_filename, _, _ in processed_files[split]:
                f.write(f"{base_filename}\n")
    
    
    print("\nDataset preparation complete!")
    print(f"Total images in train set: {len(processed_files['train'])}")
    print(f"Total images in validation set: {len(processed_files['val'])}")
    print(f"Total images in test set: {len(processed_files['test'])}")
    print(f"Output directory: {os.path.abspath(config.OUTPUT_BASE_DIR)}")

if __name__ == "__main__":
    prepare_dataset()