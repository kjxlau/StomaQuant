import os
import json
import argparse
import numpy as np
from PIL import Image

def get_class_names(class_file):
    with open(class_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def calculate_polygon_area(points):
    """Calculates area using Shoelace formula and ensures it is a standard float."""
    x = np.array(points[0::2])
    y = np.array(points[1::2])
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area)

def yolo_segmentation_to_coco(image_dir, label_dir, class_names):
    coco_output = {
        "info": {"description": "Converted from YOLO Segmentation", "year": 2023},
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Categories (COCO standard is usually 1-indexed, but 0-indexed works if consistent)
    for i, class_name in enumerate(class_names):
        coco_output["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "object",
        })

    image_id_counter = 1
    annotation_id_counter = 1
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(img_extensions):
            continue

        image_path = os.path.join(image_dir, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        coco_output["images"].append({
            "id": image_id_counter,
            "file_name": filename,
            "width": int(width),
            "height": int(height),
        })

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    class_id = int(parts[0])
                    raw_coords = [float(p) for p in parts[1:]]
                    
                    # --- CRITICAL FIX: CHECK IF NORMALIZED ---
                    # If any value is > 1.1, we assume the data is already in pixels
                    is_normalized = max(raw_coords) <= 1.1
                    
                    poly_denorm = []
                    for i in range(0, len(raw_coords), 2):
                        if is_normalized:
                            px = raw_coords[i] * width
                            py = raw_coords[i+1] * height
                        else:
                            px = raw_coords[i]
                            py = raw_coords[i+1]
                        poly_denorm.append(float(px))
                        poly_denorm.append(float(py))

                    if len(poly_denorm) < 6: continue

                    # Bounding Box calculation
                    poly_np = np.array(poly_denorm).reshape(-1, 2)
                    x_min, y_min = np.min(poly_np, axis=0)
                    x_max, y_max = np.max(poly_np, axis=0)
                    bw = x_max - x_min
                    bh = y_max - y_min
                    
                    area = calculate_polygon_area(poly_denorm)

                    coco_output["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": class_id,
                        "segmentation": [poly_denorm],
                        "area": area,
                        "bbox": [float(x_min), float(y_min), float(bw), float(bh)],
                        "iscrowd": 0,
                    })
                    annotation_id_counter += 1
            
        image_id_counter += 1

    return coco_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--class_file", required=True)
    parser.add_argument("--output_file", default="fixed_annotations.json")
    args = parser.parse_args()

    class_names = get_class_names(args.class_file)
    coco_data = yolo_segmentation_to_coco(args.image_dir, args.label_dir, class_names)

    with open(args.output_file, 'w') as f:
        # indent=4 makes the file readable and validates JSON structure
        json.dump(coco_data, f, indent=4)

    print(f"Done! Saved {len(coco_data['annotations'])} annotations.")

if __name__ == "__main__":
    main()
