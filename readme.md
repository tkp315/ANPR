## Introduction

This project focuses on image labeling and training a YOLO model for object detection tasks. By following the steps outlined below, you can effectively prepare your dataset and train a YOLO model.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Pip (Python package manager)

## Creating a Virtual Environment

To create a virtual environment for your project, follow these steps:

1. Open your terminal or command prompt.

2. Run the following command to create a virtual environment:
   ```bash
   python -m venv <name_of_virtual_environment>
   <name_of_virtual_environment>\Scripts\activate
   ```

## Capturing Images

Captured images through cv2 lib and saved in folder named test_images

## Labeling Images

Labeling tool
pip install labelimg
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py

In the LabelImg app, open the directory containing your test_images.
label the part of images you want to train
this generates a txt file for YOLO to train on that specific dataset
saved inside test_images folder

## Folder Structures

modelImages/
├── images/
│ ├── training/
│ └── validation/
└── labels/
├── training/
└── validation/
Inside the training and validation folders, place all the corresponding images and their labeled files.

## Prepration for Model Training

Once you have labeled your images and organized them, you can proceed to train your model in Google Colab.

Upload your modelImages folder to Google Drive to access it from Colab.
because there we can run on gpu so our model will train quickly as compared to on CPU
Open Google Colab and set up your notebook for training with YOLO.

## Training the YOLO Model 
1. Install the Ultralytics YOLO package in your Colab notebook:
!pip install ultralytics

Use your labeled data and follow the training instructions specific to YOLO to train your model.

!yolo task=detect mode=train model=yolov8s.pt data=/content/datasets/freedomtech/data.yaml epochs=100 imgsz=800 plots=True

After training, the output model file (e.g., best.pt) will be generated, download it and use in the main.py.

## Detecting the model 
