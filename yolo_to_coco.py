import os
import json
import argparse
import numpy as np
from PIL import Image

def get_class_names(class_file):
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def calculate_polygon_area(points):
    """
    Calculates the area of a polygon using the Shoelace formula.
    points: [x1, y1, x2, y2, ...]
    """
    x = points[0::2]
    y = points[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def yolo_segmentation_to_coco(image_dir, label_dir, class_names):
    coco_output = {
        "info": {"description": "Converted from YOLO Segmentation"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, class_name in enumerate(class_names):
        coco_output["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "object",
        })

    image_id_counter = 0
    annotation_id_counter = 0

    # Supporting common image formats
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_dir, filename)
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

            image_info = {
                "id": image_id_counter,
                "file_name": filename,
                "width": width,
                "height": height,
            }
            coco_output["images"].append(image_info)

            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 7: # A polygon needs at least 3 points (1 class + 6 coords)
                            continue
                            
                        class_id = int(parts[0])
                        poly_norm = [float(p) for p in parts[1:]]
                        
                        # De-normalize coordinates
                        poly_denorm = []
                        for i in range(0, len(poly_norm), 2):
                            poly_denorm.append(poly_norm[i] * width)
                            poly_denorm.append(poly_norm[i+1] * height)

                        # Calculate precise area using Shoelace formula
                        area = calculate_polygon_area(poly_denorm)

                        # COCO still requires a bounding box [x_min, y_min, width, height]
                        # even if you are doing segmentation.
                        poly_np = np.array(poly_denorm).reshape(-1, 2)
                        x_min, y_min = np.min(poly_np, axis=0)
                        x_max, y_max = np.max(poly_np, axis=0)
                        bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

                        annotation_info = {
                            "id": annotation_id_counter,
                            "image_id": image_id_counter,
                            "category_id": class_id,
                            "segmentation": [poly_denorm], # This retains the polygon
                            "area": float(area),           # Precise polygon area
                            "bbox": bbox,                  # Required for COCO compatibility
                            "iscrowd": 0,
                        }
                        coco_output["annotations"].append(annotation_info)
                        annotation_id_counter += 1
            
            image_id_counter += 1

    return coco_output

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation to COCO.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--class_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="annotations.json")
    args = parser.parse_args()

    class_names = get_class_names(args.class_file)
    coco_data = yolo_segmentation_to_coco(args.image_dir, args.label_dir, class_names)

    with open(args.output_file, 'w') as f:
        json.dump(coco_data, f)

    print(f"Done! Saved to {args.output_file}")

if __name__ == "__main__":
    main()
