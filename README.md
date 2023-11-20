# 🛳️ Ship Detector

## 📝 Overview
This project aims to locate ships in images using a U-Net convolutional neural network. The U-Net architecture is known for its effectiveness in image segmentation tasks.

## ⚙️ Installation

1. Clone the repository:
```
git clone https://github.com/MaksymZhytnikov/ShipDetection.git
cd ShipDetection
```

2. Install dependencies:
```
pip install -r requirements.txt
```


## 💾 Data
Dataset contains images with ships. Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

The `train_ship_segmentations_v2.csv` file provides the ground truth (in run-length encoding format) for the training images.

You can download data [here](https://www.kaggle.com/competitions/airbus-ship-detection/data). 
Place the downloaded data into the `data` folder.

## 👨🏻‍🎨 Data Exploration

Data Exploration is present in the `data_analysis.ipynb`
The data exploration section involves plotting images with specified amount of ships, analyzing the distribution of ships in images, filtering images, and balancing the dataset for training.

## 🏃🏻‍♂️ Training Pipeline
The training pipeline includes loading the dataset, balancing it, creating data generators, defining the U-Net model, and training the model with the specified parameters.

### 🧬 U-Net Architecture
The U-Net architecture is employed for ship detection. The model is compiled with a custom dice loss and dice coefficient as evaluation metric.

### 👩🏻‍🔬 Data Augmentation
Training data is augmented using various transformations such as rotation, width and height shifts, zoom, and flips.

### ⚖️ Callbacks
Callbacks are implemented to monitor and save the best model weights based on validation dice coefficient. Additionally, learning rate reduction and early stopping are employed for efficient training.

### 🚀 Start Training

The model was trained on 5 epochs due to **lack of computing resources**.
You can change the number of epochs and other hyperparameters to fit better model.

To train the model, run the following command:
```
python training_pipeline.py
```

## 🔮 Inference Pipeline

The inference pipeline involves selecting an image with a specific number of ships, resizing the image and ground truth masks, and making predictions using the loaded model.

### 🚀 Usage

To launch the inference pipeline run the next command in your terminal:

```
streamlit run inference_pipeline.py
```
