# Stomata Detection
A dataset comprising of 450 microscopy images were gathered from the adaxial and abaxial surfaces of barley, sugarcane and rice leaves. Barley and sugarcane leaf images were captured at 200x magnification, while rice leaf images were taken at 400x. Stomata within these images were manually annotated using the LabelMe polygon tool, and the resulting JSON annotations were transformed into binary mask formats. These annotated images and corresponding masks served as training data for the YOLOv12 and RF-DETR object detection model. Dataset was split into 80% training, 15% validation and 5% for testing. 

## 1. Manual Annotation of Images with LabelMe
Download and install app from https://github.com/wkentaro/labelme
<br> Using the polygon tool, draw polygon objects encircling the stomata (object of interest) and label it with a class name "stomata". </br>
A JSON annotation file will be created for the class object which contains coordinates information of the polygon objects in 2D space in the corresponding image when you hit the 'save' button.

<img width="622" height="464" alt="labelme" src="https://github.com/user-attachments/assets/32442cec-06da-446b-a9e8-9fcefe2cdec0" />

## 2. JSON to Binary Mask script
Run python [json_to_binary_mask.py](https://github.com/kjxlau/stomata/blob/main/json_to_binary_mask.py) 
<br> To convert JSON coordinate index files into binary mask in PNG format </br>

## 3. Convert segmented binary mask into YOLO format and initiate YOLOv12 model training
Run python [train_yolov12_model.py](https://github.com/kjxlau/stomata/blob/main/train_yolov12_model.py)
<br> Specify the number of epochs for training and augmentation techniques to use. </br>
We can try 300 epochs for this stomata project, batch size of 4.
Augmentation includes scaling, flipping left-right, flipping up-down and rotating at various angles for training.
<br> [Image Augmentation Docs](https://docs.roboflow.com/datasets/dataset-versions/image-augmentation) </br>
Arrange your image and annotation files as follows:
```
/path/to/your/dataset/
├── data.yaml
├── images
│   ├── train
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── val
│       ├── image3.jpg
│       ├── image4.png
│       └── ...
└── labels
    ├── train
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val
        ├── image3.txt
        ├── image4.txt
        └── ...
```

YOLOv12 integrates Flash Attention to significantly reduce computation time during training and inference. This optimization is especially effective on modern GPU architectures, including Turing: NVIDIA T4, Quadro RTX series, Ampere: RTX 30 series, A30, A40, A100, Ada Lovelace: RTX 40 series, Hopper: H100, H200. While GPU acceleration is highly recommended for optimal performance, YOLOv12 remains compatible with CPU-based training. However, users should expect longer runtimes when operating without GPU support.

<img width="1346" height="814" alt="image" src="https://github.com/user-attachments/assets/67dd1075-3305-4364-9a0b-9e0898933e9f" />

## 4. Image analysis and stomatal traits inference
<img width="1667" height="929" alt="image" src="https://github.com/user-attachments/assets/a2940670-67a4-474c-8138-48c1aafa27c5" />

## 5. Convert YOLO annotations to COCO format
Execute python [yolo_to_coco.py](https://github.com/kjxlau/stomata/blob/main/yolo_to_coco.py) 
<br>/path/to/images/train /path/to/labels/train /path/to/data.yaml /path/to/output/_annotations.coco.json </br>

## 6. Image Augmentation for RF-DETR training
Execute python 
Unlike YOLOv12, image augmentation is not build into the RF-DETR pipeline. Therefore, augmentation must be performed as a preliminary step.

## 7. Train RF-DETR model to compare performance against YOLOv12
Run python [train_RF_DETR.py](https://github.com/kjxlau/stomata/blob/main/train_RF_DETR.py) to train model for 300 epochs, batch size of 4. 
<br>Arrange your files such that the _annotations.coco.json is in the same folder as your images to train. </br>
```
/path/to/your/dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── image5.jpg
    ├── image6.jpg
    └── ...
```
