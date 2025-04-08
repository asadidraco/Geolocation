import os
import json
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse
from skimage import measure


class Config:
    
    MODEL_PATH = "best_model.pth"
    ENCODER = "resnet34"
    
    
    CATEGORIES_PATH = "dataset/categories.json"
    IMAGE_SIZE = 512
    
    
    INPUT_DIR = None  
    OUTPUT_PRED_DIR = "Target"
    
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes):
    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=None,
        classes=num_classes,
        activation=None,
    )
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()
    return model

def get_transform():
    return A.Compose([
        A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def create_colormap(n_classes):
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    return colors

def apply_colormap(mask, colors):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(1, len(colors)):  
        colored_mask[mask == class_idx] = colors[class_idx]
    return colored_mask

def predict_image(model, image_path, transform, num_classes):
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    
    
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(Config.DEVICE)
    
    
    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()
    
    
    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8), 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    
    return pred_mask

def efficient_smoothing(mask, min_size=100):

    smoothed = mask.copy()
    
    
    smoothed = cv2.medianBlur(smoothed, 5)
    
    
    for class_id in np.unique(smoothed):
        if class_id == 0:  
            continue
        
        
        binary = (smoothed == class_id).astype(np.uint8)
        
        
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        
        
        for prop in props:
            if prop.area < min_size:
                
                region_mask = (labels == prop.label)
                
                
                smoothed[region_mask] = 0
    
    
    for class_id in np.unique(smoothed):
        if class_id == 0:
            continue
        
        
        binary = (smoothed == class_id).astype(np.uint8)
        
        
        blurred = cv2.GaussianBlur(binary, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)
        
        
        smoothed[binary == 1] = 0
        smoothed[thresholded == 1] = class_id
    
    return smoothed

def save_predictions(args):
    
    Config.INPUT_DIR = args.input_dir
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    Config.OUTPUT_PRED_DIR = args.output_dir
    
    
    categories_path = args.categories_path or Config.CATEGORIES_PATH
    if os.path.exists(categories_path):
        with open(categories_path, 'r') as f:
            categories = json.load(f)
        num_classes = len(categories) + 1  
    else:
        print(f"Warning: Categories file {categories_path} not found")
        print("Using default of 6 classes (including background)")
        categories = {str(i): f"Class_{i}" for i in range(1, 6)}
        num_classes = 6
    
    
    model_path = args.model_path or Config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path, num_classes)
    
    
    transform = get_transform()
    
    
    colors = create_colormap(num_classes)
    
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        image_files.extend(
            [os.path.join(Config.INPUT_DIR, f) for f in os.listdir(Config.INPUT_DIR) 
             if f.lower().endswith(ext)]
        )
    
    if not image_files:
        print(f"No image files found in {Config.INPUT_DIR}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    
    for image_path in tqdm(image_files, desc="Generating colored predictions"):
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        
        pred_mask = predict_image(model, image_path, transform, num_classes)
        
        if pred_mask is None:
            continue
        
        
        print(f"Processing {base_filename} - Shape: {pred_mask.shape}, Classes: {np.unique(pred_mask)}")
        
        
        try:
            smoothed_mask = efficient_smoothing(pred_mask)
            
            colored_pred = apply_colormap(smoothed_mask, colors)
        except Exception as e:
            print(f"Error smoothing {base_filename}: {e}")
            
            colored_pred = apply_colormap(pred_mask, colors)
        
        
        output_pred_path = os.path.join(Config.OUTPUT_PRED_DIR, f"{base_filename}.png")
        cv2.imwrite(output_pred_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
    
    print(f"Colored predictions saved to {os.path.abspath(Config.OUTPUT_PRED_DIR)}")
    print(f"Total images processed: {len(image_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save colored predictions")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="Target", help="Directory to save colored predictions")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file")
    parser.add_argument("--categories_path", type=str, default=None, help="Path to categories JSON file")
    parser.add_argument("--min_size", type=int, default=100, help="Minimum region size to keep")
    
    args = parser.parse_args()
    
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
    else:
        save_predictions(args)