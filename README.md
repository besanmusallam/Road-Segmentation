# Road Segmentation Using Satellite Images

This repository contains the implementation of road segmentation using satellite images, leveraging state-of-the-art deep learning models such as U-Net and DeepLabv3. The project explores binary segmentation techniques to extract road networks from satellite imagery, with applications in navigation, urban planning, and disaster response.

---

## Repository Structure

```
DL_project/u-net
├── baseline_unet
│   ├── model/                # Saved models
│   ├── unet/                 # U-Net model architecture
│   ├── requirements.txt      # Dependencies
│   ├── train.py              # Training script
│   ├── unet_test.ipynb       # U-Net testing notebook
│   └── vis.ipynb             # Visualization notebook
├── DeepLab
│   └── deep_labv3+_enhanced_Final.ipynb  # DeepLabv3+ implementation notebook
├── preprocessing
│   ├── data_loader.py        # Data loading and augmentation script
│   └── sattelite_segmentation.ipynb  # Preprocessing and segmentation pipeline
└── README.md                 # Project documentation
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/besanmusallam/Road-Segmentation.git
```

### 2. Install Dependencies
Ensure you have Python installed (>=3.8) and then run:
```bash
pip install -r baseline_unet/requirements.txt
```

---

## Usage

### Preprocessing
To preprocess the data, use the `data_loader.py` script as follows:
```python
# Import the data_loader
from data_loader import load_data, ImageDataGenerator

# Load the data
images, masks = load_data(image_dir, mask_dir)

# Create DataGenerator instances
train_generator = ImageDataGenerator(images, masks, target_size=(512, 512), batch_size=32, augment=True, save_dir=save_dir, preprocessed_save_dir=preprocess_dir)

# Run the generator and save the data
for _ in range(len(train_generator)):
    _ = train_generator[_]

print("Data preprocessing and augmentation completed.")
```
Ensure `image_dir`, `mask_dir`, `save_dir`, and `preprocess_dir` are appropriately defined paths.

### Training the U-Net Model
Navigate to the `baseline_unet` directory and execute:
```bash
python train.py
```
This script trains the U-Net model and saves the trained model in the `model/` directory.

### Running the DeepLabv3+ Notebook
Open the `DeepLab/deep_labv3+_enhanced_Final.ipynb` notebook and execute the cells in sequence to train or test the DeepLabv3+ model.

---

## Results
- **U-Net:**
  - Achieved a training F1-score of 0.77 and testing F1-score of 0.75.
- **DeepLabv3+:**
  - Achieved a training F1-score of 0.91 and testing F1-score of 0.86.

Training of U-Net can be found in `baseline_unet/unet_test.ipynb` .
Visualization of U-Net can be found in `baseline_unet/vis.ipynb` .
Training and visualization of DeepLab V3+ can ba found in `DeepLab/deep_labv3+_enhanced_Final.ipynb` .

---

## Dataset
The dataset used in this project is provided by the EPFL ML Road Segmentation challenge. You can find more details and download the dataset from the following link:

[EPFL ML Road Segmentation Challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)

Ensure you have downloaded the dataset and placed it in the appropriate directory structure before running the scripts.

---

## Acknowledgments
- The University of Jordan for support and resources.
- Dr. Tamam ALsarhan for her continuous support.
- AIcrowd platform for providing the dataset.

