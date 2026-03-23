import os
import json
import argparse
import numpy as np

def calculate_polygon_area(points):
    """Calculates the area of a polygon using the Shoelace formula."""
    x = points[0::2]
    y = points[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def labelme_to_coco(label_dir, class_file, output_file):
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    category_map = {name: i for i, name in enumerate(class_names)}

    coco_output = {
        "info": {"description": "Converted from LabelMe JSON"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(class_names)]
    }

    ann_id_counter = 0
    img_id_counter = 0

    # Get list of files and skip the output file itself if it exists
    files = [f for f in os.listdir(label_dir) if f.endswith(".json") and f != os.path.basename(output_file)]

    for filename in files:
        json_path = os.path.join(label_dir, filename)
        
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {filename}: Not a valid JSON file.")
                continue

        # SAFEGUARD: Check if this is actually a LabelMe file
        if 'shapes' not in data:
            print(f"Skipping {filename}: No 'shapes' key found (not a LabelMe file).")
            continue

        # Image Info
        width = data.get('imageWidth')
        height = data.get('imageHeight')
        image_name = data.get('imagePath', filename.replace('.json', '.jpg'))

        image_info = {
            "id": img_id_counter,
            "file_name": image_name,
            "width": width,
            "height": height,
        }
        coco_output["images"].append(image_info)

        # Process Shapes
        for shape in data['shapes']:
            label = shape['label']
            if label not in category_map:
                print(f"Warning: Label '{label}' in {filename} not found in class file. Skipping shape.")
                continue
            
            points = np.array(shape['points'])
            if len(points) < 3: # Need at least 3 points for a polygon
                continue

            segmentation = points.flatten().tolist()

            # Bounding Box [x_min, y_min, width, height]
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

            area = calculate_polygon_area(segmentation)

            annotation = {
                "id": ann_id_counter,
                "image_id": img_id_counter,
                "category_id": category_map[label],
                "segmentation": [segmentation],
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0,
            }
            coco_output["annotations"].append(annotation)
            ann_id_counter += 1

        img_id_counter += 1

    return coco_output

def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON to COCO format.")
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--class_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="test_coco.json")
    args = parser.parse_args()

    coco_data = labelme_to_coco(args.label_dir, args.class_file, args.output_file)

    with open(args.output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"\nDone! Processed {len(coco_data['images'])} images.")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()
