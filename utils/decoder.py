
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import random
from scipy.spatial.distance import cdist
import gc
import uuid
# Import ILORE components directly
from ilore.util import vector2dict, neuclidean
from ilore.explanation import ImageExplanation
from ilore.ineighgen import ImageRandomAdversarialGeneratorLatent
from ilore.decision_tree import learn_local_decision_tree
from ilore.rule import get_rule, get_counterfactual_rules, Condition, Rule, apply_counterfactual
import time 


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvNextDecoder(nn.Module):
    def __init__(self):
        super(ConvNextDecoder, self).__init__()
        self.upsample_blocks = nn.ModuleList([

            nn.Sequential(
                nn.ConvTranspose2d(1024, 768, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm([768, 16, 16]),
                nn.GELU()
            ),

            nn.Sequential(
                nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm([512, 32, 32]),
                nn.GELU()
            ),

            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm([256, 64, 64]),
                nn.GELU()
            ),

            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm([128, 128, 128]),
                nn.GELU()
            ),

            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm([64, 256, 256]),
                nn.GELU()
            )
        ])

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, encoder_features=None):
        for idx, block in enumerate(self.upsample_blocks):
            x = block(x)
            if encoder_features is not None and idx < len(encoder_features):
                x += encoder_features[-(idx+1)]
        x = self.final_conv(x)
        return self.tanh(x)
    

class Discriminator(nn.Module):
    def __init__(self, bottleneck_dim=1024):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(bottleneck_dim, 768, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([768, 8, 8]),

            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([512, 8, 8]),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([256, 8, 8]),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([128, 8, 8]),

            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([64, 8, 8])
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, latent_vectors):
        x = self.conv_layers(latent_vectors)
        x = self.fc_layers(x)
        return x

def extract_encoder_features(encoder, images):
    features = []
    x = images
    for layer in encoder.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            features.append(x)
    return features, x

class AutoencoderWrapper:
    def __init__(self, encoder, bottleneck, decoder, discriminator=None, latent_dim=1024*8*8):
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
        self.latent_shape = (1, 1024, 8, 8)
        
        self.original_images = {}
        self.original_encoder_features = {}
    
    def encode(self, images):
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=0)
                
            if images.shape[-1] == 3:
                images = np.transpose(images, (0, 3, 1, 2))
                
            images_tensor = torch.tensor(images, dtype=torch.float32).to(device)
        else:
            images_tensor = images
        
        with torch.no_grad():
            for i in range(images_tensor.shape[0]):
                img_hash = hash(images_tensor[i].cpu().numpy().tobytes())
                if img_hash not in self.original_images:
                    self.original_images[img_hash] = images_tensor[i:i+1].clone()
                    
                    features, _ = extract_encoder_features(self.encoder, images_tensor[i:i+1])
                    self.original_encoder_features[img_hash] = features
            
            _, encoder_output = extract_encoder_features(self.encoder, images_tensor)
            
            latent = self.bottleneck(encoder_output)
            
            if latent.dim() > 2:
                self.latent_shape = latent.shape
                
            latent_flat = latent.reshape(latent.size(0), -1).cpu().numpy()
            
            return latent_flat
    
    def decode(self, latent_vectors, original_images=None):

        if isinstance(latent_vectors, np.ndarray):
            latent_vectors = torch.tensor(latent_vectors, dtype=torch.float32)
            
        batch_size = latent_vectors.shape[0]
        max_batch_size = 16
        
        all_results = []
        for i in range(0, batch_size, max_batch_size):
            batch = latent_vectors[i:i+max_batch_size].to(device)
            
            # Reshape if necessary
            if len(batch.shape) == 2:
                current_batch_size = batch.shape[0]
                if hasattr(self, 'latent_shape'):
                    channels, height, width = self.latent_shape[1:]
                    batch = batch.reshape(current_batch_size, channels, height, width)
                else:
                    # Default shape if not determined
                    batch = batch.reshape(current_batch_size, 1024, 8, 8)
            
            with torch.no_grad():
                encoder_features = None
                
                if i == 0 and batch.shape[0] > 0:
                    found = False
                    
                    if original_images is not None:
                        if isinstance(original_images, np.ndarray):
                            if original_images.shape[-1] == 3:  # HWC format
                                original_images = np.transpose(original_images, (0, 3, 1, 2))
                            original_tensor = torch.tensor(original_images, dtype=torch.float32).to(device)
                        else:
                            original_tensor = original_images.to(device)
                            
                        encoder_features, _ = extract_encoder_features(self.encoder, original_tensor[:current_batch_size])
                        found = True
                    
                    if not found and batch.shape[0] > 0:
                        for img_hash, features in self.original_encoder_features.items():
                            encoder_features = features
                            found = True
                            break
                
                if encoder_features is None:
                    dummy_input = torch.zeros(current_batch_size, 3, 256, 256).to(device)
                    encoder_features, _ = extract_encoder_features(self.encoder, dummy_input)
                
                # Decode
                decoded = self.decoder(batch, encoder_features)
                
                all_results.append(decoded.cpu().numpy())
                
            torch.cuda.empty_cache()
            
        decoded_images = np.concatenate(all_results, axis=0) if all_results else np.array([])
        
        if decoded_images.shape[1] == 3:
            decoded_images = np.transpose(decoded_images, (0, 2, 3, 1))
            
        return decoded_images