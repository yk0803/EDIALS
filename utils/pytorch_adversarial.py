
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

class PyTorchImageRandomAdversarialGeneratorLatent(ImageRandomAdversarialGeneratorLatent):
    
    def generate_latent(self):
        while True:
            lz_img = np.random.normal(size=(1, self.autoencoder.latent_dim))
            
            if self.autoencoder.discriminator is not None and self.valid_thr > 0.0:
                tensor_input = torch.tensor(lz_img, dtype=torch.float32).to(device)
                
                if hasattr(self.autoencoder, 'latent_shape'):
                    tensor_input = tensor_input.reshape(self.autoencoder.latent_shape)
                else:
                    # Default shape
                    tensor_input = tensor_input.reshape(1, 1024, 8, 8)
                
                # Run discriminator
                with torch.no_grad():
                    disc_out = self.autoencoder.discriminator(tensor_input)
                    
                    # Get scalar value
                    if isinstance(disc_out, torch.Tensor):
                        discriminator_out = disc_out.item() if disc_out.numel() == 1 else disc_out.flatten()[0].item()
                    else:
                        discriminator_out = disc_out
                
                if discriminator_out > self.valid_thr:
                    return lz_img
            else:
                return lz_img
                
    def generate(self, img, num_samples=1000):
        
        lZ_img = self.generate_latent_samples(num_samples)
        lZ_img[0] = self.autoencoder.encode(np.array([img]))[0]
        
        Z_img = self.autoencoder.decode(lZ_img)
        Z_img[0] = img.copy()  # Ensure original image is kept
        
        Yb = self.bb_predict(Z_img)
        class_value = Yb[0]
        
        lZ_img, Z_img = self._balance_neigh(lZ_img, Z_img, Yb, num_samples, class_value)
        Yb = self.bb_predict(Z_img)
        
        Z = np.array(lZ_img)
        return Z, Yb, class_value