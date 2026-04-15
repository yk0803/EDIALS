import os
from PIL import Image
from matplotlib import transforms
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.decoder import AutoencoderWrapper
from torchvision import models, transforms
from utils.generate_iterative import generate_adversarial_with_ilore
from utils.decoder import ConvNextDecoder
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main function to run the ILORE-ABELE integration
def run_ilore_abele_iterative(black_box_path='black_box.pt', 
                               autoencoder_path='AAE_2.pt',
                               dataset_dir='Dataset TV/Train',
                               output_dir="distorted_images",
                               num_classes=None,
                               max_iterations=20,
                               distortion_factor=0.5,
                               num_samples=240):
    
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    
    class FaceDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.class_to_idx = {}
            self.idx_to_class = {}
            
            class_folders = [folder for folder in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, folder))]
            
            class_folders.sort()
            
            for idx, class_folder in enumerate(class_folders):
                self.class_to_idx[class_folder] = idx
                self.idx_to_class[idx] = class_folder
            
            for class_folder in class_folders:
                class_path = os.path.join(root_dir, class_folder)
                class_id = self.class_to_idx[class_folder]
                
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(class_id)
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path

    print(f"Loading dataset from {dataset_dir}")
    train_dataset = FaceDataset(root_dir=dataset_dir, transform=transform)
    
    print(f"Loading black box from {black_box_path}")
    if num_classes is None:
        num_classes = len(train_dataset.class_to_idx)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Create model and load weights
    black_box = models.convnext_base(weights=None)
    black_box.classifier[2] = nn.Linear(1024, num_classes)
    black_box.load_state_dict(torch.load(black_box_path, map_location=device))
    black_box = black_box.to(device)
    black_box.eval()
    
    autoencoder_dict = torch.load(autoencoder_path, map_location=device, weights_only=False)
        
    if isinstance(autoencoder_dict, dict) and 'encoder' in autoencoder_dict:
        encoder = autoencoder_dict['encoder'].to(device)
        bottleneck = autoencoder_dict['bottleneck'].to(device)
        decoder = autoencoder_dict['decoder'].to(device)
        discriminator = autoencoder_dict['discriminator'].to(device)

    autoencoder_wrapper = AutoencoderWrapper(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        discriminator=discriminator,
        latent_dim=1024 * 8 * 8  # 1024 channels x 8x8 spatial dimensions
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all class folders upfront
    print("Created output folders for all classes...")
    for class_idx in range(num_classes):
        class_name = train_dataset.idx_to_class.get(class_idx, f"class_{class_idx}")
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
    
    batch_size = 1
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    stats = {
        'total': 0,
        'successful': 0,
        'unsuccessful': 0,
        'total_iterations': 0,
        'mse_values': [],
        'psnr_values': []
    }
    for images, labels, paths in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = black_box(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(images)):
            img = images[i]
            label = labels[i]
            path = paths[i]
            
            class_name = train_dataset.idx_to_class[label.item()]
            
            class_output_dir = os.path.join(output_dir, class_name)
            
            filename = os.path.basename(path)
            # Generate adversarial example
            distorted_image, iterations, is_misclassified = generate_adversarial_with_ilore(
                image=img,
                label=label,
                black_box=black_box,
                autoencoder_wrapper=autoencoder_wrapper,
                output_dir=class_output_dir,
                max_iterations=max_iterations,
                distortion_factor=distortion_factor,
                num_samples=num_samples,
                original_filename=filename,
                num_classes=240
            )
            # print(is_misclassified)

            stats['total'] += 1
            if is_misclassified:
                stats['successful'] += 1
                stats['total_iterations'] += iterations
                
                if distorted_image is not None:
                    mse = F.mse_loss(img, distorted_image).item()
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100.0
                    stats['mse_values'].append(mse)
                    stats['psnr_values'].append(psnr)
            else:
                stats['unsuccessful'] += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if stats['successful'] > 0:
        stats['avg_iterations'] = stats['total_iterations'] / stats['successful']
        stats['success_rate'] = stats['successful'] / stats['total'] * 100
        stats['avg_mse'] = sum(stats['mse_values']) / len(stats['mse_values'])
        stats['avg_psnr'] = sum(stats['psnr_values']) / len(stats['psnr_values'])
    
    print("\nAdversarial Example Generation Complete")
    print(f"Total examples processed: {stats['total']}")
    print(f"Successfully misclassified: {stats['successful']} ({stats.get('success_rate', 0):.2f}%)")
    print(f"Average iterations: {stats.get('avg_iterations', 0):.2f}")
    
    if stats['successful'] > 0:
        print(f"Average MSE: {stats.get('avg_mse', 0):.6f}")
        print(f"Average PSNR: {stats.get('avg_psnr', 0):.2f} dB")
    
    stats_path = os.path.join(output_dir, "generation_stats.txt")
    try:
        with open(stats_path, 'w') as f:
            f.write("ABELE-ILORE Adversarial Example Generation Statistics\n")
            f.write("==================================================\n\n")
            f.write(f"Total examples processed: {stats['total']}\n")
            f.write(f"Successfully misclassified: {stats['successful']} ({stats.get('success_rate', 0):.2f}%)\n")
            f.write(f"Failed attempts: {stats['unsuccessful']}\n")
            f.write(f"Average iterations for successful examples: {stats.get('avg_iterations', 0):.2f}\n")
            
            if stats['successful'] > 0:
                f.write(f"Average MSE: {stats.get('avg_mse', 0):.6f}\n")
                f.write(f"Average PSNR: {stats.get('avg_psnr', 0):.2f} dB\n")
                
            f.write("\nParameters:\n")
            f.write(f"Max iterations: {max_iterations}\n")
            f.write(f"Distortion factor: {distortion_factor}\n")
            f.write(f"Neighborhood samples: {num_samples}\n")
    except Exception as e:
        print(f"Error saving statistics to file: {e}")
    
    return stats
