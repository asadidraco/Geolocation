import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse


class Config:
    
    MODEL_PATH = "model_outputs/best_model.pth"
    ENCODER = "resnet34"
    
    
    CATEGORIES_PATH = "dataset/categories.json"
    IMAGE_SIZE = 512
    
    
    INPUT_DIR = None  
    OUTPUT_DIR = "inference_results"
    
    
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

def predict_image(model, image_path, transform, num_classes):
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return None, None
    
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
    
    return image, pred_mask

def create_colormap(n_classes):
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    return colors

def process_images(args):
    
    Config.INPUT_DIR = args.input_dir
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    Config.OUTPUT_DIR = args.output_dir
    
    
    if os.path.exists(Config.CATEGORIES_PATH):
        with open(Config.CATEGORIES_PATH, 'r') as f:
            categories = json.load(f)
        num_classes = len(categories) + 1  
    else:
        print(f"Warning: Categories file {Config.CATEGORIES_PATH} not found")
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
    
    
    for image_path in tqdm(image_files):
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        
        image, pred_mask = predict_image(model, image_path, transform, num_classes)
        
        if image is None or pred_mask is None:
            continue
        
        
        colored_pred = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for class_idx in range(1, num_classes):  
            colored_pred[pred_mask == class_idx] = colors[class_idx]
        
        
        overlay = image.copy()
        mask_indices = pred_mask > 0
        overlay[mask_indices] = overlay[mask_indices] * 0.5 + colored_pred[mask_indices] * 0.5
        
        
        plt.figure(figsize=(20, 5))
        
        
        plt.subplot(1, 4, 1)
        plt.title("IMAGE")
        plt.imshow(image)
        plt.axis("off")
        
        
        
        plt.subplot(1, 4, 2)
        plt.title("GT")
        
        blank_gt = np.zeros_like(colored_pred)
        plt.imshow(blank_gt)
        plt.axis("off")
        
        
        plt.subplot(1, 4, 3)
        plt.title("PREDICTED")
        plt.imshow(colored_pred)
        plt.axis("off")
        
        
        plt.subplot(1, 4, 4)
        plt.title("OVERLAY")
        plt.imshow(overlay)
        plt.axis("off")
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f"{base_filename}_result.png"), dpi=150)
        plt.close()
    
    
    plt.figure(figsize=(6, 4))
    for i, (cat_id, cat_name) in enumerate(categories.items()):
        color = colors[int(cat_id)]
        plt.bar(i, 1, color=color/255)
        plt.text(i, 0.5, f"{cat_id}: {cat_name}", 
                 ha='center', va='center', rotation=90, 
                 color='white' if sum(color) < 1.5*255 else 'black')
    
    plt.title("Class Legend")
    plt.xlim(-0.5, len(categories) - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "class_legend.png"))
    plt.close()
    
    print(f"Results saved to {os.path.abspath(Config.OUTPUT_DIR)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on new images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save results")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file")
    parser.add_argument("--ground_truth_dir", type=str, default=None, help="Optional directory containing ground truth masks")
    
    args = parser.parse_args()
    
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
    else:
        process_images(args)