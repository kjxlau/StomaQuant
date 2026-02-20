import os
import random
import yaml
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import torch

def set_seeds():
    SEED_VALUE = 42
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
set_seeds()

def get_all_data(data_dir):
    """
    Returns a list of dictionaries. Each dictionary contains:
    {
      'img': path_to_jpg,
      'stoma': path_to_mask_or_None,
      'pore': path_to_mask1_or_None
    }
    """
    data_samples = []

    for root, _, files in os.walk(data_dir):
        # Filter for JPG images first
        jpg_files = [f for f in files if f.endswith('.jpg')]
        
        for file in jpg_files:
            img_path = os.path.join(root, file)
            
            # Construct expected mask paths
            stoma_path = os.path.join(root, file.replace('.jpg', '_mask.png'))
            pore_path = os.path.join(root, file.replace('.jpg', '_mask1.png'))
            
            has_stoma = os.path.exists(stoma_path)
            has_pore = os.path.exists(pore_path)

            # CRITICAL FIX: 
            # Include image if AT LEAST ONE mask exists.
            if has_stoma or has_pore:
                data_samples.append({
                    'img': img_path,
                    'stoma': stoma_path if has_stoma else None,
                    'pore': pore_path if has_pore else None
                })
            else:
                # Only warn if NO masks exist at all (unlabeled image)
                # print(f"Skipping {file} - No masks found.")
                pass

    # Sort to ensure reproducibility
    data_samples.sort(key=lambda x: x['img'])
    
    return data_samples

def mask_to_polygons(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
            poly = contour.reshape(-1).tolist()
            # Ensure valid polygon (at least 3 points = 6 coordinates)
            if len(poly) > 4: 
                polygons.append(poly)
    return polygons

def process_batch(batch_data, output_images_dir, output_labels_dir):
    
    for sample in batch_data:
        img_path = sample['img']
        stoma_path = sample['stoma']
        pore_path = sample['pore']
        
        # 1. Copy Image
        img = cv2.imread(img_path)
        if img is None: continue
        
        file_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(output_images_dir, file_name))
        
        height, width = img.shape[:2]
        txt_filename = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(output_labels_dir, txt_filename)

        with open(txt_path, 'w') as file_object:
            
            # --- PROCESS CLASS 0: STOMATA (If exists) ---
            if stoma_path is not None:
                mask_stoma = cv2.imread(stoma_path, cv2.IMREAD_UNCHANGED)
                if mask_stoma is not None:
                    unique_values = np.unique(mask_stoma)
                    for value in unique_values:
                        if value == 0: continue
                        
                        object_mask = (mask_stoma == value).astype(np.uint8) * 255
                        polygons = mask_to_polygons(object_mask)
                        
                        for poly in polygons:
                            normalized_poly = [format(coord / width if i % 2 == 0 else coord / height, '.6f') for i, coord in enumerate(poly)]
                            file_object.write(f"0 " + " ".join(normalized_poly) + "\n")

            # --- PROCESS CLASS 1: PORE (If exists) ---
            if pore_path is not None:
                mask_pore = cv2.imread(pore_path, cv2.IMREAD_UNCHANGED)
                if mask_pore is not None:
                    unique_values = np.unique(mask_pore)
                    for value in unique_values:
                        if value == 0: continue
                        
                        object_mask = (mask_pore == value).astype(np.uint8) * 255
                        polygons = mask_to_polygons(object_mask)
                        
                        for poly in polygons:
                            normalized_poly = [format(coord / width if i % 2 == 0 else coord / height, '.6f') for i, coord in enumerate(poly)]
                            file_object.write(f"1 " + " ".join(normalized_poly) + "\n")

def create_yaml(output_yaml_path, train_path, val_path, test_path, nc=2):
    names = ['stomata', 'pore']
    
    yaml_data = {
        'names': names,
        'nc': nc,
        'train': train_path,
        'val': val_path,
        'test': test_path
    }
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

def yolo_dataset_preparation():
    data_dir = './inputs'
    output_dir = 'stomaquant_pore2'
 
    # Directory Setup
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    test_images_dir = os.path.join(output_dir, 'test', 'images')
    
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    test_labels_dir = os.path.join(output_dir, 'test', 'labels')
 
    for d in [train_images_dir, val_images_dir, test_images_dir, train_labels_dir, val_labels_dir, test_labels_dir]:
        os.makedirs(d, exist_ok=True)
 
    # 1. Get ALL data (Permissive Mode)
    # returns list of dicts: [{'img':..., 'stoma':..., 'pore':...}, ...]
    all_data = get_all_data(data_dir)
    
    print(f"Total images found with at least one mask: {len(all_data)}")

    if len(all_data) == 0:
        print("No labeled images found. Check input path.")
        return

    # 2. Splitting Logic
    # Train: 70%, Val: 15%, Test: 15%
    train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 3. Process Data
    print("Processing Train...")
    process_batch(train_data, train_images_dir, train_labels_dir)
    
    print("Processing Val...")
    process_batch(val_data, val_images_dir, val_labels_dir)
    
    print("Processing Test...")
    process_batch(test_data, test_images_dir, test_labels_dir)
    
    # 4. Create YAML
    output_yaml_path = os.path.join(output_dir, 'data.yaml')
    create_yaml(output_yaml_path, 
                os.path.abspath(train_images_dir), 
                os.path.abspath(val_images_dir), 
                os.path.abspath(test_images_dir))
    
    print("Dataset preparation complete.")
 
if __name__ == "__main__":
    yolo_dataset_preparation()
