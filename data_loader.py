import os
import cv2
import math 
import numpy as np
import random
from tensorflow.keras.utils import Sequence

# Function to load data
def load_data(image_dir, mask_dir, img_size=(400, 400)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Ensure alignment
    assert len(image_files) == len(mask_files), "Number of images and masks do not match!"
    for img, mask in zip(image_files, mask_files):
        assert img.split('.')[0].split('_')[-1] == mask.split('.')[0].split('_')[-1], f"Mismatch: {img} and {mask}"

    # Load and preprocess
    images = [cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_GRAYSCALE) for f in image_files]
    masks = [cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in mask_files]

    return np.array(images), np.array(masks)

# Custom ImageDataGenerator class without Data Augmentation
class ImageDataGenerator(Sequence):
    def __init__(self, images, masks, target_size=(512, 512), batch_size=32, augment=False):
        self.images = images
        self.masks = masks
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        # Use math.ceil to include all samples
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        # Get a batch of images and masks
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.images))  # Handle last batch
        batch_images = self.images[start:end]
        batch_masks = self.masks[start:end]

        # Preprocess the batch
        return self._preprocess_batch(batch_images, batch_masks)

    def _preprocess_batch(self, batch_images, batch_masks):
        processed_images = []
        processed_masks = []

        for img, mask in zip(batch_images, batch_masks):
            # Resize the image and mask to the target size
            resized_img = cv2.resize(img, self.target_size)
            resized_mask = cv2.resize(mask, self.target_size)

            if self.augment:  # Apply data augmentation if augment is True
                augmented_img, augmented_mask = self._augment(resized_img, resized_mask)
                processed_images.append(augmented_img / 255.0)
                processed_masks.append(augmented_mask / 255.0)
            else:
                processed_images.append(resized_img / 255.0)
                processed_masks.append(resized_mask / 255.0)

        return np.array(processed_images), np.array(processed_masks)

    def _augment(self, image, mask):
        # Example augmentations applied to both image and mask
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Random rotation
        angle = random.randint(-30, 30)
        rows, cols = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, matrix, (cols, rows))
        mask = cv2.warpAffine(mask, matrix, (cols, rows))

        return image, mask