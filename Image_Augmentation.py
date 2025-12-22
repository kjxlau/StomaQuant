import albumentations as A
import cv2
import json
import os
from tqdm import tqdm

# --- 1. Define Augmentation Pipeline ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=1.0, border_mode=cv2.BORDER_CONSTANT),
    A.RandomSizedBBoxSafeCrop(width=640, height=640, erosion_rate=0.2, p=1.0),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

# --- 2. Configuration ---
SOURCE_DIR = "./stomata/test/"
OUTPUT_DIR = "./stomata_augmented/test/"
NUM_AUGMENTATIONS_PER_IMAGE = 3

# --- 3. Create Output Directories ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 4. Load Original Annotations ---
ann_path = os.path.join(SOURCE_DIR, '_annotations.coco.json')
with open(ann_path, 'r') as f:
    coco_data = json.load(f)

# --- 5. Prepare New Annotation Structure (THE FIX IS HERE) ---
new_coco_data = {
    # Add placeholder info and licenses keys to ensure compliance
    "info": {
        "description": "Augmented Stomata Dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "Augmentation Script",
    },
    "licenses": [
        {
            "id": 1,
            "name": "Unknown",
            "url": ""
        }
    ],
    "images": [],
    "annotations": [],
    "categories": coco_data['categories'] # Copy categories from original file
}
new_image_id = 0
new_annotation_id = 0

# --- 6. Augmentation Loop ---
print("Generating augmented dataset...")
for image_info in tqdm(coco_data['images']):
    # (The rest of the loop is the same as before)
    image_path = os.path.join(SOURCE_DIR, image_info['file_name'])
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Skipping missing image {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_id = image_info['id']
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    bboxes = [ann['bbox'] for ann in anns]
    labels = [ann['category_id'] for ann in anns]

    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
            
            new_filename = f"{os.path.splitext(image_info['file_name'])[0]}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, new_filename), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))

            new_coco_data['images'].append({
                "id": new_image_id,
                "file_name": new_filename,
                "width": transformed['image'].shape[1],
                "height": transformed['image'].shape[0]
            })

            for bbox, label in zip(transformed['bboxes'], transformed['class_labels']):
                new_coco_data['annotations'].append({
                    "id": new_annotation_id,
                    "image_id": new_image_id,
                    "category_id": label,
                    "bbox": [round(x, 2) for x in bbox], # Round bbox values
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                new_annotation_id += 1
            
            new_image_id += 1
        except Exception as e:
            print(f"Skipping augmentation for {image_info['file_name']} due to error: {e}")

# --- 7. Save the New Annotation File ---
output_ann_path = os.path.join(OUTPUT_DIR, '_annotations.coco.json')
with open(output_ann_path, 'w') as f:
    json.dump(new_coco_data, f, indent=4)

print(f"\nDone! Augmented dataset created at {OUTPUT_DIR}")
