import gc
import uuid
import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from utils.ilorem import ILOREM
from utils.decoder import extract_encoder_features
from torchvision import models, transforms
import uuid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_adversarial_with_ilore(image, label, black_box, autoencoder_wrapper, 
                                  output_dir="distorted_images", 
                                  max_iterations=20, 
                                  distortion_factor=0.5,
                                  num_samples=200,
                                  original_filename=None,
                                  num_classes=None):

    
    def bb_predict(images):
        if isinstance(images, np.ndarray):
            if images.shape[-1] == 3:
                images = np.transpose(images, (0, 3, 1, 2))  # Convert to channels first (CHW)
            images = torch.tensor(images, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = black_box(images)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()

    original_class = label.item()
    # print(f"Original class (from label): {original_class}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    class_folders = {idx: f"class_{idx}" for idx in range(num_classes)}

    initial_prediction = bb_predict(image.unsqueeze(0))[0]
    # print(f"Initial prediction: {initial_prediction}")

    if initial_prediction != original_class:
        print(f"Image is already misclassified as {initial_prediction} (original class: {original_class}). Saving without processing.")
        
        # Map the original and predicted classes to their respective folder names
        original_class_folder = class_folders.get(original_class, f"class_{original_class}")
        predicted_class_folder = class_folders.get(initial_prediction, f"class_{initial_prediction}")
        
        # Define the path for saving
        save_path = os.path.join(output_dir, f"original_misclassified_{original_class}_as_{initial_prediction}_{uuid.uuid4()}.png")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        to_pil = transforms.ToPILImage()
        original_pil = to_pil(image.cpu())
        
        original_pil.save(save_path, format='PNG', compress_level=0)
        print(f"Saved original image to: {save_path}")
        
        return image, 0, True
        
    img_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    
    num_classes = black_box.classifier[2].out_features
    class_values = list(range(num_classes))
    
    explainer = ILOREM(
        bb_predict=bb_predict, 
        class_name='class', 
        class_values=class_values,
        neigh_type='rnd',
        autoencoder=autoencoder_wrapper,
        use_rgb=True,
        verbose=True
    )
    
    # Generate explanation
    # print(f"Generating explanation with {num_samples} samples...")
    explanation = explainer.explain_instance(img_np, num_samples=num_samples)
    
    # Print explanation details
    # print(f"Black box prediction: {explanation.bb_pred}")
    # print(f"Decision tree prediction: {explanation.dt_pred}")
    # print(f"Factual rule: {explanation.rule}")
    # print(f"Found {len(explanation.crules)} counterfactual rules")
    # print(f"Fidelity: {explanation.fidelity:.4f}")
    
    # If no counterfactual rules, save original image and return
    if not explanation.crules or len(explanation.crules) == 0:
        print("No counterfactual rules found. Saving original image.")
        
        save_path = os.path.join(output_dir, f"original_no_rules_{original_class}.png")
        
        to_pil = transforms.ToPILImage()
        original_pil = to_pil(image.cpu())
        
        original_pil.save(save_path, format='PNG', compress_level=0)
        print(f"Saved original image to: {save_path}")
        
        return image, 0, False
    
    latent = autoencoder_wrapper.encode(image.unsqueeze(0).cpu().numpy())[0]
    
    current_latent = latent.copy()
    iterations = 0
    is_misclassified = False
    
    with torch.no_grad():
        original_encoder_features, _ = extract_encoder_features(autoencoder_wrapper.encoder, image.unsqueeze(0))
    
    original_brightness = torch.mean(image).item()
    
    original_latent_norm = np.linalg.norm(latent)
    
    rule_idx = 0
    crule = explanation.crules[rule_idx]
    delta = explanation.deltas[rule_idx]
    
    all_features = []
    for rule_idx in range(len(explanation.crules)):
        crule = explanation.crules[rule_idx]
        delta = explanation.deltas[rule_idx]
        
        # print(f"Using rule {rule_idx}: {crule}")
        
        for condition in delta:
            feature_idx = int(condition.att.split("_")[1])
            # Add to all_features if not already there
            if not any(f[0] == feature_idx for f in all_features):
                all_features.append((feature_idx, condition))

    max_features = len(all_features)

    feature_magnitudes = []
    for feat_idx, condition in all_features:
        magnitude = abs(latent[feat_idx])
        feature_magnitudes.append((feat_idx, condition, magnitude))


    sorted_features = sorted(feature_magnitudes, key=lambda x: x[2], reverse=True)
    ordered_features = [(feat_idx, condition) for feat_idx, condition, _ in sorted_features]
    features = [(feat_idx, condition) for feat_idx, condition, _ in sorted_features]


    # If no features, can't proceed
    if max_features == 0:
        print("No features to distort in rules. Saving original image.")
        save_path = os.path.join(output_dir, f"original_no_features_{original_class}.png")
        to_pil = transforms.ToPILImage()
        original_pil = to_pil(image.cpu())
        original_pil.save(save_path, format='PNG', compress_level=0)
        print(f"Saved original image to: {save_path}")
        return image, 0, False

    # distortion_levels = {feat_idx: 0 for feat_idx, _ in all_features}
    distortion_levels = {feat_idx: 0 for feat_idx, _ in features} 

    # features = all_features
        
    # Keep track of the best (misclassified) result
    best_distorted_image = None
    best_distortion_metrics = None
    best_prediction = None
    
    for iteration in range(max_iterations):
        current_latent_norm = np.linalg.norm(current_latent)
        if current_latent_norm > 0:  # Avoid division by zero
            scaling_factor = original_latent_norm / current_latent_norm
            current_latent = current_latent * scaling_factor
            
        latent_tensor = torch.tensor(current_latent.reshape(1, -1), dtype=torch.float32)
        
        if hasattr(autoencoder_wrapper, 'latent_shape'):
            shape = autoencoder_wrapper.latent_shape
            latent_tensor = latent_tensor.reshape(1, shape[1], shape[2], shape[3])
        else:
            latent_tensor = latent_tensor.reshape(1, 1024, 8, 8)
        
        latent_tensor = latent_tensor.to(device)
        
        with torch.no_grad():
            current_image = autoencoder_wrapper.decoder(latent_tensor, original_encoder_features)
            
            current_brightness = torch.mean(current_image.squeeze(0)).item()
            if current_brightness > 0:  # Avoid division by zero
                brightness_ratio = original_brightness / current_brightness
                # Limit brightness correction to avoid extreme values
                brightness_ratio = max(min(brightness_ratio, 2.0), 0.5)  
                # Apply brightness correction
                current_image = torch.clamp(current_image * brightness_ratio, 0, 1)
        
        current_pred = bb_predict(current_image)[0]
        
        current_mse = F.mse_loss(image.unsqueeze(0).to(device), current_image).item()
        current_psnr = 20 * np.log10(1.0 / np.sqrt(current_mse)) if current_mse > 0 else 100.0
        
        # Update best result if misclassified
        if current_pred != original_class:
            if best_distorted_image is None or current_mse < best_distortion_metrics['MSE']:
                best_distorted_image = current_image.clone()
                best_distortion_metrics = {'MSE': current_mse, 'PSNR': current_psnr}
                best_prediction = current_pred
            
            is_misclassified = True
            # print(f"Image misclassified after {iteration} iterations!")
            break
        
        # print(f"\nIteration {iteration+1}/{max_iterations}:")
        # print(f"Current prediction: {current_pred}")
        # print(f"Current distortion levels: {distortion_levels}")
        # print(f"Current MSE: {current_mse:.6f}, PSNR: {current_psnr:.2f} dB")
        
        if iteration < max_features:
            # First N iterations: Add one new feature per iteration
            feature_to_increment = features[iteration][0]
            distortion_levels[feature_to_increment] = 1
            # print(f"Adding feature {feature_to_increment} with level 1")
        else:
            cycle_position = (iteration - max_features) % max_features
            feature_to_increment = features[cycle_position][0]
            distortion_levels[feature_to_increment] += 1
            # print(f"Incrementing feature {feature_to_increment} to level {distortion_levels[feature_to_increment]}")
        
        # Apply the appropriate distortion to each feature
        features_modified = []
        for feat_idx, condition in features:
            level = distortion_levels[feat_idx]
            if level > 0:
                # Apply distortion based on the condition and level
                if condition.op == ">":
                    current_latent[feat_idx] += distortion_factor
                    features_modified.append(f"{feat_idx}^{level}")
                else:  # op is "<="
                    current_latent[feat_idx] -= distortion_factor
                    features_modified.append(f"{feat_idx}^{level}")
        
        # print(f"Modified features: {' AND '.join(features_modified)}")
        
        iterations += 1
        
        # Periodically force garbage collection to manage memory
        if iteration % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Use the best result if misclassified, otherwise use the last image
    if best_distorted_image is not None:
        final_image = best_distorted_image.squeeze(0)
        mse = best_distortion_metrics['MSE']
        psnr = best_distortion_metrics['PSNR']
        final_pred = best_prediction
    else:
        final_image = current_image.squeeze(0)
        mse = current_mse
        psnr = current_psnr
        final_pred = current_pred
    
    # print(f"\nResults:")
    # print(f"MSE: {mse:.6f}")
    # print(f"PSNR: {psnr:.2f} dB")
    # print(f"Is misclassified: {is_misclassified}")
    
    # Define the output path based on whether misclassification was successful
    if is_misclassified:
        save_path = os.path.join(output_dir, f"adversarial_{original_class}_to_{final_pred}_{uuid.uuid4()}.png")
        status = "misclassified"
    else:
        save_path = os.path.join(output_dir, f"distorted_but_still_{original_class}_{uuid.uuid4()}.png")
        status = "still_correctly_classified"
    
    # Convert tensor to PIL image
    to_pil = transforms.ToPILImage()
    distorted_pil = to_pil(final_image.cpu())
    
    # Save image
    distorted_pil.save(save_path, format='PNG', compress_level=0)
    # print(f"Saved {status} image to: {save_path}")
    
    
    return final_image, iterations, is_misclassified
