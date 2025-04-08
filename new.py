import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class Config:
    DATASET_DIR = "dataset"
    IMAGE_SIZE = 512
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    OUTPUT_DIR = "model_outputs"


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        inter_channels = max(1, in_channels // reduction)

        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cse = self.channel_excitation(x) * x
        sse = self.spatial_se(x) * x
        return cse + sse


class MAnetWithSCSE(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=1):
        super(MAnetWithSCSE, self).__init__()

        self.manet = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=None
        )
        if 'resnet34' in encoder_name:
            encoder_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        self.encoder_scse = nn.ModuleList([SCSEModule(ch) for ch in encoder_channels])

    def forward(self, x):
        return self.manet(x)


class PreparedSegmentationDataset(Dataset):
    def __init__(self, dataset_dir, split, transform=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        file_list_path = os.path.join(dataset_dir, f'{split}_files.txt')
        with open(file_list_path, 'r') as f:
            self.file_basenames = [line.strip() for line in f]
        with open(os.path.join(dataset_dir, 'categories.json'), 'r') as f:
            self.categories = json.load(f)
        self.num_classes = len(self.categories) + 1  # +1 for background
    
    def __len__(self):
        return len(self.file_basenames)
    
    def __getitem__(self, idx):
        basename = self.file_basenames[idx]
        image_path = os.path.join(self.dataset_dir, self.split, 'images', f'{basename}.png')
        mask_path = os.path.join(self.dataset_dir, self.split, 'masks', f'{basename}.png')
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        return image, mask.long()


def get_training_transform():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_validation_transform():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def train_model():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    
    train_dataset = PreparedSegmentationDataset(config.DATASET_DIR, 'train', get_training_transform())
    val_dataset = PreparedSegmentationDataset(config.DATASET_DIR, 'val', get_validation_transform())
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    
    model = MAnetWithSCSE(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=train_dataset.num_classes
    ).to(config.DEVICE)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}") as pbar:
            for images, masks in pbar:
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                
                preds = torch.argmax(outputs, dim=1)
                iou = calculate_iou(preds, masks, train_dataset.num_classes)
                val_iou += iou
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best_model.pth"))
            patience_counter = 0
            print(f"New best model saved with val_loss: {val_loss:.4f}, val_iou: {val_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'history': history
        }, os.path.join(config.OUTPUT_DIR, "latest_checkpoint.pth"))
    
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Validation IoU', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "training_history.png"))
    plt.close()
    
    return model, history


def calculate_iou(preds, targets, num_classes, ignore_index=255):
    """Calculate Intersection over Union (IoU) for each class"""
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    
    valid = (targets != ignore_index)
    preds = preds[valid]
    targets = targets[valid]
    
    for class_id in range(num_classes):
        pred_inds = (preds == class_id)
        target_inds = (targets == class_id)
        
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        
        if union == 0:
            ious.append(float('nan'))  
        else:
            ious.append(float(intersection) / float(union))
    
    return np.nanmean(ious)


def evaluate_on_test_set(model_path=None):
    config = Config()
    
    
    test_dataset = PreparedSegmentationDataset(
        dataset_dir=config.DATASET_DIR,
        split='test',
        transform=get_validation_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    
    model = MAnetWithSCSE(
        encoder_name=config.ENCODER,
        encoder_weights=None,
        classes=test_dataset.num_classes
    )
    
    
    model_path = model_path or os.path.join(config.OUTPUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct_pixels = 0
    total_pixels = 0
    class_iou = {}
    class_accuracy = {}
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating on test set"):
            images = images.to(config.DEVICE)
            masks = masks.long().to(config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            test_loss += loss.item()
            
            
            pred_masks = torch.argmax(outputs, dim=1)
            correct_pixels += (pred_masks == masks).sum().item()
            total_pixels += masks.numel()
            
            
            for class_id in range(test_dataset.num_classes):
                pred_class = (pred_masks == class_id)
                true_class = (masks == class_id)
                
                
                class_correct = (pred_class & true_class).sum().item()
                class_total = true_class.sum().item()
                
                if class_id not in class_accuracy:
                    class_accuracy[class_id] = {'correct': 0, 'total': 0}
                
                class_accuracy[class_id]['correct'] += class_correct
                class_accuracy[class_id]['total'] += class_total
                
                
                intersection = (pred_class & true_class).sum().float().item()
                union = (pred_class | true_class).sum().float().item()
                
                if union > 0:
                    if class_id not in class_iou:
                        class_iou[class_id] = {'intersection': 0, 'union': 0}
                    
                    class_iou[class_id]['intersection'] += intersection
                    class_iou[class_id]['union'] += union
    
    
    test_loss /= len(test_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    
    mean_iou_per_class = {}
    for class_id in class_iou:
        mean_iou_per_class[class_id] = class_iou[class_id]['intersection'] / class_iou[class_id]['union']
    mean_iou = np.mean(list(mean_iou_per_class.values()))
    
    
    mean_accuracy_per_class = {}
    for class_id in class_accuracy:
        if class_accuracy[class_id]['total'] > 0:
            mean_accuracy_per_class[class_id] = class_accuracy[class_id]['correct'] / class_accuracy[class_id]['total']
        else:
            mean_accuracy_per_class[class_id] = float('nan')
    
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    
    print("\nPer-class metrics:")
    print(f"{'Class':<10}{'Accuracy':<12}{'IoU':<10}")
    for class_id in range(test_dataset.num_classes):
        acc = mean_accuracy_per_class.get(class_id, float('nan'))
        iou = mean_iou_per_class.get(class_id, float('nan'))
        print(f"{class_id:<10}{acc:.4f}{'':<8}{iou:.4f}")
    
    
    metrics = {
        'test_loss': test_loss,
        'pixel_accuracy': pixel_accuracy,
        'mean_iou': mean_iou,
        'class_iou': mean_iou_per_class,
        'class_accuracy': mean_accuracy_per_class
    }
    
    with open(os.path.join(config.OUTPUT_DIR, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def visualize_predictions(model_path=None, num_samples=5, save_dir=None):
    config = Config()
    save_dir = save_dir or config.OUTPUT_DIR
    
    
    test_dataset = PreparedSegmentationDataset(
        dataset_dir=config.DATASET_DIR,
        split='test',
        transform=get_validation_transform()
    )
    
    
    model = MAnetWithSCSE(
        encoder_name=config.ENCODER,
        encoder_weights=None,
        classes=test_dataset.num_classes
    )
    
    
    model_path = model_path or os.path.join(config.OUTPUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    
    n_classes = test_dataset.num_classes
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    
    
    plt.figure(figsize=(18, 5 * len(indices)))
    
    for i, idx in enumerate(indices):
        image, mask = test_dataset[idx]
        basename = test_dataset.file_basenames[idx]
        
        
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(config.DEVICE)
            pred_mask = model(image_tensor)
            pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()
        
        
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        
        
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_pred = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        
        for class_idx in range(n_classes):  
            colored_mask[mask == class_idx] = colors[class_idx]
            colored_pred[pred_mask == class_idx] = colors[class_idx]
        
        
        plt.subplot(len(indices), 3, i*3 + 1)
        plt.title(f"Image: {basename}")
        plt.imshow(image)
        plt.axis("off")
        
        plt.subplot(len(indices), 3, i*3 + 2)
        plt.title("Ground Truth")
        plt.imshow(colored_mask)
        plt.axis("off")
        
        plt.subplot(len(indices), 3, i*3 + 3)
        plt.title("Prediction")
        plt.imshow(colored_pred)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_predictions.png"))
    plt.close()
    
    
    os.makedirs(os.path.join(save_dir, "prediction_samples"), exist_ok=True)
    for i, idx in enumerate(indices):
        image, mask = test_dataset[idx]
        basename = test_dataset.file_basenames[idx]
        
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(config.DEVICE)
            pred_mask = model(image_tensor)
            pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()
        
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_pred = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        
        for class_idx in range(n_classes):
            colored_mask[mask == class_idx] = colors[class_idx]
            colored_pred[pred_mask == class_idx] = colors[class_idx]
        
        
        comparison = np.hstack([image, colored_mask, colored_pred])
        cv2.imwrite(os.path.join(save_dir, "prediction_samples", f"{basename}_comparison.png"), 
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, using CPU")
    
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("Starting training...")
    model, history = train_model()
    
    print("\nEvaluating on test set...")
    metrics = evaluate_on_test_set()
    
    print("\nVisualizing predictions...")
    visualize_predictions(num_samples=5)