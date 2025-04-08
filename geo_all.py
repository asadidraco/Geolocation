import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from tqdm import tqdm
import timm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class AerialPatchDataset(Dataset):
    def __init__(self, gt_dir, pred_dir, transform=None, patches_per_image=35, min_size=228, max_size=512):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.transform = transform
        self.patches_per_image = patches_per_image
        self.min_size = min_size
        self.max_size = max_size
        
        
        if not os.path.exists(gt_dir):
            raise ValueError(f"Ground truth directory does not exist: {gt_dir}")
        if not os.path.exists(pred_dir):
            raise ValueError(f"Prediction directory does not exist: {pred_dir}")
        
        
        self.image_files = [f for f in os.listdir(gt_dir) if f.endswith(('.jpg', '.png', '.tif', '.jpeg'))]
        
        
        valid_image_files = []
        for img_file in self.image_files:
            gt_path = os.path.join(gt_dir, img_file)
            pred_path = os.path.join(pred_dir, img_file)
            
            if not os.path.exists(pred_path):
                print(f"Warning: Image {img_file} exists in ground truth but not in prediction dir. Skipping.")
                continue
            
            try:
                
                with Image.open(gt_path) as img:
                    pass
                with Image.open(pred_path) as img:
                    pass
                valid_image_files.append(img_file)
            except Exception as e:
                print(f"Error opening image {img_file}: {e}. Skipping.")
        
        self.image_files = valid_image_files
        print(f"Found {len(self.image_files)} valid image pairs")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No valid image pairs found in {gt_dir} and {pred_dir}")
        
    def __len__(self):
        return len(self.image_files) * self.patches_per_image
    
    def __getitem__(self, idx):
        image_idx = idx // self.patches_per_image
        image_filename = self.image_files[image_idx]
        
        gt_image_path = os.path.join(self.gt_dir, image_filename)
        pred_image_path = os.path.join(self.pred_dir, image_filename)
        
        try:
            gt_image = Image.open(gt_image_path).convert('RGB')
            pred_image = Image.open(pred_image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_filename}: {e}")
            
            image_filename = self.image_files[0]
            gt_image_path = os.path.join(self.gt_dir, image_filename)
            pred_image_path = os.path.join(self.pred_dir, image_filename)
            gt_image = Image.open(gt_image_path).convert('RGB')
            pred_image = Image.open(pred_image_path).convert('RGB')
        
        original_width, original_height = gt_image.size
        
        patch_size = self.min_size
        
        max_x = original_width - patch_size
        max_y = original_height - patch_size
        
        if max_x <= 0 or max_y <= 0:
            new_width = max(original_width, patch_size + 100)
            new_height = max(original_height, patch_size + 100)
            gt_image = gt_image.resize((new_width, new_height))
            pred_image = pred_image.resize((new_width, new_height))
            original_width, original_height = gt_image.size
            max_x = original_width - patch_size
            max_y = original_height - patch_size
        
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        
        patch = pred_image.crop((x, y, x + patch_size, y + patch_size))
        
        
        center_x = (x + patch_size // 2) / original_width
        center_y = (y + patch_size // 2) / original_height
        
        
        metadata = {
            'original_width': original_width,
            'original_height': original_height,
            'patch_x': x,
            'patch_y': y,
            'patch_size': patch_size,
            'image_path': gt_image_path
        }
        
        
        if self.transform:
            full_image_tensor = self.transform(gt_image)
            patch_tensor = self.transform(patch)
        else:
            to_tensor = transforms.ToTensor()
            full_image_tensor = to_tensor(gt_image)
            patch_tensor = to_tensor(patch)
        
        
        to_tensor = transforms.ToTensor()
        original_full_image = to_tensor(gt_image)
        original_patch = to_tensor(patch)
        
        center = torch.tensor([center_x, center_y], dtype=torch.float32)
        
        return full_image_tensor, patch_tensor, center, original_full_image, original_patch, metadata

def custom_collate(batch):
    full_images = []
    patches = []
    centers = []
    orig_full_images = []
    orig_patches = []
    metadata_list = []
    
    for item in batch:
        full_images.append(item[0])
        patches.append(item[1])
        centers.append(item[2])
        orig_full_images.append(item[3])
        orig_patches.append(item[4])
        metadata_list.append(item[5])
    
    full_images = torch.stack(full_images)
    patches = torch.stack(patches)
    centers = torch.stack(centers)
    orig_full_images = torch.stack(orig_full_images)
    orig_patches = torch.stack(orig_patches)
    
    return full_images, patches, centers, orig_full_images, orig_patches, metadata_list


class DeepFeatureTemplateMatching(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True):
        super(DeepFeatureTemplateMatching, self).__init__()
        # Reference MAP
        if backbone == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048

        elif backbone == "resnet18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 512

        elif backbone == "efficientnet_b4":
            base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = base_model.features
            self.feature_dim = 1792

        elif backbone == "convnext_base":
            base_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 1024

        elif backbone == "swin_b":

            base_model = timm.create_model("swin_base_patch4_window7_224", pretrained=pretrained, features_only=True)
            self.feature_extractor = nn.Sequential(*base_model)
            self.feature_dim = self.feature_extractor[-1].out_channels

        elif backbone == "vit_b_16":
            base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            def forward_features(x):
                n = x.shape[0]
                x = base_model._process_input(x)  # Patch embedding
                cls_token = base_model.cls_token.expand(n, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                x = base_model.encoder(x)
                x = x[:, 1:]  # remove cls token
                h = int((x.shape[1]) ** 0.5)
                x = x.permute(0, 2, 1).reshape(n, 768, h, h)
                return x
            self.feature_extractor = forward_features
            self.feature_dim = 768

        elif backbone == "efficientnet_v2_m":
            base_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = base_model.features
            self.feature_dim = 1280

        elif backbone == "densenet201":
            base_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
            self.feature_dim = 1920

        elif backbone == "mobilenet_v3_large":
            base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = base_model.features
            self.feature_dim = 960

        elif backbone == "beit_base_patch16_224":
            base_model = timm.create_model("beit_large_patch16_512", pretrained=pretrained)
            if not hasattr(base_model, 'pos_embed') or base_model.pos_embed is None:
                base_model.pos_embed = nn.Parameter(torch.zeros(1, base_model.patch_embed.num_patches + 1, base_model.embed_dim))
            class BeitWrapper(nn.Module):
                def __init__(self, beit):
                    super().__init__()
                    self.beit = beit

                def forward(self, x):
                    n = x.shape[0]
                    x = self.beit.patch_embed(x)
                    cls_token = self.beit.cls_token.expand(n, -1, -1)
                    x = torch.cat((cls_token, x), dim=1)
                    x = self.beit.pos_drop(x + self.beit.pos_embed)
                    for blk in self.beit.blocks:
                        x = blk(x)
                    x = x[:, 1:]
                    h = int((x.shape[1]) ** 0.5)
                    x = x.permute(0, 2, 1).reshape(n, x.shape[2], h, h)
                    return x
            self.feature_extractor = BeitWrapper(base_model)
            self.feature_dim = 1024

        elif backbone == "vit_base_patch16_224":
            base_model = timm.create_model("vit_base_patch16_512", pretrained=pretrained, features_only=True)
            self.feature_extractor = nn.Sequential(*list(base_model.children()))
            self.feature_dim = base_model.feature_info.channels()[-1]

        elif backbone == "convnextv2_base":
            base_model = timm.create_model("convnextv2_base", pretrained=pretrained, features_only=True)
            self.feature_extractor = nn.Sequential(*list(base_model.children()))
            self.feature_dim = base_model.feature_info.channels()[-1]
        
        elif backbone == "mobilevitv2_150":
            base_model = timm.create_model("mobilevitv2_150.fb_in1k", pretrained=pretrained, features_only=True)
            self.feature_extractor = nn.Sequential(*list(base_model.children()))
            self.feature_dim = base_model.feature_info.channels()[-1]

            
        self.patch_projection = nn.Conv2d(self.feature_dim, 256, kernel_size=1)
        self.image_projection = nn.Conv2d(self.feature_dim, 256, kernel_size=1)
        
        self.correlation_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        self.activation = nn.GELU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.feature_map_size = 16
        
        flattened_size = 64 * (self.feature_map_size//4) * (self.feature_map_size//4)
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, full_image, patch):
        image_features = self.feature_extractor(full_image)
        patch_features = self.feature_extractor(patch)
        
        image_features = self.image_projection(image_features)
        patch_features = self.patch_projection(patch_features)
        
        correlation_map = self.compute_correlation(image_features, patch_features)
        
        x = self.correlation_conv(correlation_map.unsqueeze(1))
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        predicted_coords = self.fc2(x)
        
        return predicted_coords, correlation_map
    
    def compute_correlation(self, image_features, patch_features):
        batch_size = patch_features.size(0)
        patch_vector = torch.mean(patch_features, dim=(2, 3))
        
        patch_vector = patch_vector.view(batch_size, -1, 1, 1)
        
        correlation = torch.sum(image_features * patch_vector, dim=1)
        
        return correlation


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone().detach()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)

def tensor_to_numpy(tensor):
    return tensor.cpu().numpy().transpose((1, 2, 0))

def create_multi_size_visualization(model, gt_dir, output_dir="multi_size_visualization", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    
    patch_sizes = [64, 128, 192, 256, 320]
    
    image_files = [f for f in os.listdir(gt_dir) if f.endswith(('.jpg', '.png', '.tif', '.jpeg'))]
    
    
    if not image_files:
        print(f"No images found in {gt_dir}. Make sure the path is correct and contains images.")
        print(f"Files in directory: {os.listdir(gt_dir)}")
        return
        
    num_images = max(1, min(5, len(image_files)))  
    selected_images = image_files[:num_images]
    
    fig, axes = plt.subplots(len(patch_sizes), 2*num_images, figsize=(20, 15))
    
    
    if len(patch_sizes) == 1:
        axes = np.array([axes])
    if num_images == 1:
        axes = axes.reshape(len(patch_sizes), 2)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for img_idx, image_filename in enumerate(tqdm(selected_images, desc="Processing images")):
        gt_image_path = os.path.join(gt_dir, image_filename)
        
        
        if not os.path.exists(gt_image_path):
            print(f"Image {gt_image_path} does not exist, skipping...")
            continue
            
        try:
            full_image = Image.open(gt_image_path).convert('RGB')
            width, height = full_image.size
        except Exception as e:
            print(f"Error opening image {gt_image_path}: {e}")
            continue
        
        for row_idx, patch_size in enumerate(patch_sizes):
            
            center_x, center_y = width // 2, height // 2
            
            
            x = center_x - patch_size // 2
            y = center_y - patch_size // 2
            
            
            x = max(0, min(x, width - patch_size))
            y = max(0, min(y, height - patch_size))
            
            
            try:
                patch = full_image.crop((x, y, x + patch_size, y + patch_size))
                patch_path = os.path.join(output_dir, f"temp_patch_{image_filename}_{patch_size}.jpg")
                patch.save(patch_path)
            except Exception as e:
                print(f"Error cropping or saving patch: {e}")
                continue
                
            
            try:
                
                full_tensor = transform(full_image).unsqueeze(0).to(device)
                patch_tensor = transform(patch).unsqueeze(0).to(device)
                
                
                with torch.no_grad():
                    predicted_centers, correlation_maps = model(full_tensor, patch_tensor)
                
                
                pred_center_x = predicted_centers[0][0].item() * width
                pred_center_y = predicted_centers[0][1].item() * height
                
                
                error_distance = np.sqrt((center_x - pred_center_x)**2 + (center_y - pred_center_y)**2)
                error_percentage = (error_distance / np.sqrt(width**2 + height**2)) * 100
                
                
                img_col = 2 * img_idx
                patch_col = 2 * img_idx + 1
                
                
                ax_img = axes[row_idx, img_col]
                ax_img.imshow(np.array(full_image))
                ax_img.set_title(f"Full Image: {image_filename}", fontsize=8)
                
                
                rect = mpatches.Rectangle(
                    (x, y), patch_size, patch_size, 
                    linewidth=2, edgecolor='yellow', facecolor='none'
                )
                ax_img.add_patch(rect)
                
                
                ax_img.scatter(center_x, center_y, c='green', s=50, marker='o', label='Ground Truth')
                ax_img.scatter(pred_center_x, pred_center_y, c='red', s=50, marker='x', label='Prediction')
                
                
                ax_img.plot([center_x, pred_center_x], [center_y, pred_center_y], 'b-', linewidth=1.5)
                
                
                ax_img.text(
                    10, 30, 
                    f"Distance: {error_distance:.1f}px ({error_percentage:.2f}%)", 
                    bbox=dict(facecolor='white', alpha=0.7), fontsize=8
                )
                
                
                if row_idx == 0:
                    ax_img.legend(loc='upper right', fontsize=8)
                
                
                ax_patch = axes[row_idx, patch_col]
                ax_patch.imshow(np.array(patch))
                ax_patch.set_title(f"Patch ({patch_size}x{patch_size})", fontsize=8)
                
                
                if img_idx == 0:
                    ax_img.set_ylabel(f"Size: {patch_size}px", fontsize=10)
                
                
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                ax_patch.set_xticks([])
                ax_patch.set_yticks([])
                
            except Exception as e:
                print(f"Error in prediction or visualization: {e}")
                
            
            if os.path.exists(patch_path):
                os.remove(patch_path)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_size_multi_image_visualization.png"), dpi=300)
    plt.close(fig)
    
    print(f"Visualization saved to {os.path.join(output_dir, 'multi_size_multi_image_visualization.png')}")
    
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, device="cuda", visualization_dir="visualizations"):
    
    os.makedirs(visualization_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    val_distances = []
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_data in pbar:
            full_images, patches, centers = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
              predicted_centers, _ = model(full_images, patches)
              
              loss = criterion(predicted_centers, centers)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"Training Loss: {epoch_train_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        distances = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch_data in enumerate(val_pbar):
                full_images = batch_data[0].to(device)
                patches = batch_data[1].to(device)
                centers = batch_data[2].to(device)
                orig_full_images = batch_data[3]
                orig_patches = batch_data[4]
                metadata_list = batch_data[5]
                
                predicted_centers, correlation_maps = model(full_images, patches)
                
                loss = criterion(predicted_centers, centers)
                val_loss += loss.item()
                
                batch_size = centers.size(0)
                for i in range(batch_size):
                    true_center = centers[i].cpu()
                    pred_center = predicted_centers[i].cpu()
                    
                    if isinstance(metadata_list, list) and i < len(metadata_list):
                        meta = metadata_list[i]
                    else:
                        continue
                    
                    try:
                        true_x = true_center[0].item() * meta['original_width']
                        true_y = true_center[1].item() * meta['original_height']
                        pred_x = pred_center[0].item() * meta['original_width']
                        pred_y = pred_center[1].item() * meta['original_height']
                        
                        distance = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
                        diagonal = np.sqrt(meta['original_width']**2 + meta['original_height']**2)
                        normalized_distance = distance / diagonal
                        
                        distances.append(normalized_distance)
                    except (KeyError, TypeError) as e:
                        print(f"Error calculating distance for sample {i}: {e}")
                        if isinstance(meta, dict):
                            print(f"Available keys: {meta.keys()}")
                        else:
                            print(f"Metadata type: {type(meta)}")
        
        if len(val_loader) > 0:
            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                val_distances.append(avg_distance)
                print(f"Validation Loss: {epoch_val_loss:.4f}")
                print(f"Average Normalized Distance: {avg_distance:.4f} (of image diagonal)")
            else:
                print("Warning: No valid distance measurements collected")
                val_distances.append(float('nan'))
        
        
        checkpoint_path = os.path.join(visualization_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss if len(val_loader) > 0 else float('inf'),
        }, checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")
        
        
        if len(val_loader) > 0 and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = os.path.join(visualization_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"Saved best model with validation loss {best_val_loss:.4f} to {best_model_path}")
            
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:  
            multi_size_dir = os.path.join(visualization_dir, f"multi_size_epoch_{epoch+1}")
            os.makedirs(multi_size_dir, exist_ok=True)
            
            
            val_gt_dir = os.path.dirname(train_loader.dataset.gt_dir).replace('train', 'val')
            if not os.path.exists(val_gt_dir):
                print(f"Warning: Validation directory {val_gt_dir} not found. Using training directory instead.")
                val_gt_dir = train_loader.dataset.gt_dir
            
            
            if os.path.basename(train_loader.dataset.gt_dir) == 'A' and os.path.basename(val_gt_dir) != 'A':
                val_gt_dir = os.path.join(val_gt_dir, 'A')
                
            if not os.path.exists(val_gt_dir):
                print(f"Warning: Could not find valid ground truth directory at {val_gt_dir}")
                print(f"Skipping visualization for epoch {epoch+1}")
                continue
                
            print(f"Creating multi-size visualization using images from: {val_gt_dir}")
            create_multi_size_visualization(
                model=model,
                gt_dir=val_gt_dir,
                output_dir=multi_size_dir,
                device=device
            )
    
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_distances, label='Validation Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Distance')
    plt.title('Average Normalized Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'training_curves.png'))
    plt.close()
    
    return model, train_losses, val_losses, val_distances

def infer(model, gt_image_path, pred_patch_path, patch_size=None, device="cuda", visualize=True):
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    if not os.path.exists(gt_image_path):
        raise FileNotFoundError(f"Ground truth image not found: {gt_image_path}")
    if not os.path.exists(pred_patch_path):
        raise FileNotFoundError(f"Patch image not found: {pred_patch_path}")
    
    
    full_image = Image.open(gt_image_path).convert('RGB')
    patch_image = Image.open(pred_patch_path).convert('RGB')
    
    
    if patch_size is not None:
        patch_width, patch_height = patch_image.size
        if patch_width != patch_size or patch_height != patch_size:
            patch_image = patch_image.resize((patch_size, patch_size))
    
    original_width, original_height = full_image.size
    
    
    full_tensor = transform(full_image).unsqueeze(0).to(device)
    patch_tensor = transform(patch_image).unsqueeze(0).to(device)
    
    
    # with torch.no_grad():
    with torch.no_grad(), torch.cuda.amp.autocast():
        predicted_center, correlation_map = model(full_tensor, patch_tensor)
    
    
    pred_x = predicted_center[0][0].item() * original_width
    pred_y = predicted_center[0][1].item() * original_height
    
    
    if visualize:
        plt.figure(figsize=(15, 5))
        
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(full_image))
        plt.scatter(pred_x, pred_y, c='red', s=100, marker='x')
        plt.title("Full Image with Predicted Location")
        plt.axis('on')  
        
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.array(patch_image))
        plt.title(f"Query Patch ({patch_image.width}x{patch_image.height})")
        plt.axis('on')
        
        
        plt.subplot(1, 3, 3)
        correlation_np = correlation_map[0].cpu().numpy()
        
        
        if correlation_np.max() > correlation_np.min():
            normalized_correlation = (correlation_np - correlation_np.min()) / (correlation_np.max() - correlation_np.min())
        else:
            normalized_correlation = correlation_np
        
        plt.imshow(normalized_correlation, cmap='jet')
        plt.colorbar()
        plt.title("Correlation Map")
        plt.axis('on')
        
        plt.tight_layout()
        plt.savefig("inference_result.png", dpi=300)
        plt.close()
        
        print(f"Predicted center: ({pred_x:.1f}, {pred_y:.1f})")
        print(f"Visualization saved to inference_result.png")
    
    return pred_x, pred_y, correlation_map.cpu()

def main():
    DATA_DIR = "geodata"
    GT_TRAIN_DIR = os.path.join(DATA_DIR, "train", "A")
    PRED_TRAIN_DIR = os.path.join(DATA_DIR, "train", "B")
    GT_VAL_DIR = os.path.join(DATA_DIR, "val", "A")
    PRED_VAL_DIR = os.path.join(DATA_DIR, "val", "B")
    OUTPUT_DIR = "aerial_patch_output"
    VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "aerial_patch_locator.pth")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    directories = [GT_TRAIN_DIR, PRED_TRAIN_DIR, GT_VAL_DIR, PRED_VAL_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist!")
            if "train" in directory:
                print("Training directory not found. Make sure your data is organized correctly.")
                print(f"Looking for: {directory}")
                if os.path.exists(DATA_DIR):
                    print(f"Contents of {DATA_DIR}: {os.listdir(DATA_DIR)}")
                    if os.path.exists(os.path.join(DATA_DIR, 'train')):
                        print(f"Contents of train directory: {os.listdir(os.path.join(DATA_DIR, 'train'))}")
                return
    
    
    for directory in directories:
        image_count = len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.tif', '.jpeg'))])
        print(f"Found {image_count} images in {directory}")
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 35
    LEARNING_RATE = 0.0001
    PATCHES_PER_IMAGE_TRAIN = 35
    PATCHES_PER_IMAGE_VAL = 15
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    patch_size = 256
    
    train_dataset = AerialPatchDataset(
        gt_dir=GT_TRAIN_DIR,
        pred_dir=PRED_TRAIN_DIR,
        transform=transform,
        patches_per_image=PATCHES_PER_IMAGE_TRAIN,
        min_size=patch_size,
        max_size=patch_size
    )
    
    val_dataset = AerialPatchDataset(
        gt_dir=GT_VAL_DIR,
        pred_dir=PRED_VAL_DIR,
        transform=transform,
        patches_per_image=PATCHES_PER_IMAGE_VAL,
        min_size=patch_size,
        max_size=patch_size
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate
    )
    
    model = DeepFeatureTemplateMatching(backbone="densenet201", pretrained=True)
    model = model.to(device)
    
    loss_functions = {
        # "MSELoss": nn.MSELoss(),
        # "L1Loss": nn.L1Loss(),
        "SmoothL1Loss": nn.SmoothL1Loss()
    }

    for loss_name, loss_fn in loss_functions.items():
        print(f"\n=== Training with {loss_name} ===")

        model = DeepFeatureTemplateMatching(backbone="densenet201", pretrained=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        vis_dir_loss = os.path.join(VIS_DIR, loss_name)
        os.makedirs(vis_dir_loss, exist_ok=True)

        trained_model, train_losses, val_losses, val_distances = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            device=device,
            visualization_dir=vis_dir_loss
        )

        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_distances': val_distances
        }, os.path.join(OUTPUT_DIR, f"aerial_patch_model_{loss_name}.pth"))

        print(f"Model for {loss_name} saved to {os.path.join(OUTPUT_DIR, f'aerial_patch_model_{loss_name}.pth')}")

    
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_distances': val_distances
    }, MODEL_PATH)
    
    print(f"Model saved to {MODEL_PATH}")
    
    return trained_model

if __name__ == "__main__":
    main()